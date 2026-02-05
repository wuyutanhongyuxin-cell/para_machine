"""
Paradex API async client for Paradex Trader.

Features:
- Interactive Token authentication (0 fees)
- Automatic token refresh
- Request retry with exponential backoff
- Rate limit handling
- Connection pooling
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

import aiohttp

from paradex_trader.core.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    OrderError,
    InsufficientBalanceError,
)
from paradex_trader.core.models import (
    BBO,
    Orderbook,
    OrderbookLevel,
    Position,
    PositionSide,
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    AccountInfo,
)
from paradex_trader.utils.helpers import (
    exponential_backoff,
    format_size,
    safe_float,
    generate_order_id,
)

logger = logging.getLogger("paradex_trader.client")


class ParadexClient:
    """
    Asynchronous Paradex API client.

    Uses Interactive Token authentication for zero trading fees.
    Supports all necessary trading operations including:
    - Market data (BBO, orderbook, recent trades)
    - Account info (balance, positions)
    - Order management (market, limit, cancel)
    """

    def __init__(
        self,
        l2_private_key: str,
        l2_address: str,
        environment: str = "prod",
        max_retries: int = 3,
        timeout: float = 10.0,
    ):
        """
        Initialize Paradex client.

        Args:
            l2_private_key: Starknet L2 private key.
            l2_address: Starknet L2 address.
            environment: API environment ('prod' or 'testnet').
            max_retries: Maximum retry attempts for failed requests.
            timeout: Request timeout in seconds.
        """
        self.l2_private_key = l2_private_key
        self.l2_address = l2_address
        self.environment = environment
        self.max_retries = max_retries
        self.timeout = timeout

        # API configuration
        if environment == "prod":
            self.base_url = "https://api.prod.paradex.trade/v1"
        else:
            self.base_url = "https://api.testnet.paradex.trade/v1"

        # Authentication state
        self.jwt_token: Optional[str] = None
        self.jwt_expires_at: int = 0

        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        self._paradex = None  # paradex-py SDK instance

        # Rate limiting
        self._last_request_time: float = 0
        self._min_request_interval: float = 0.1  # 100ms between requests

        # Connection state
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize client and authenticate."""
        if self._initialized:
            return

        try:
            # Use ParadexSubkey for L2-only authentication
            # Paradex class requires L1 credentials, ParadexSubkey works with L2 only
            from paradex_py import Paradex
            try:
                from paradex_py import ParadexSubkey
                HAS_SUBKEY = True
            except ImportError:
                HAS_SUBKEY = False
            from paradex_py.environment import PROD, TESTNET

            env = PROD if self.environment == "prod" else TESTNET

            if HAS_SUBKEY and self.l2_private_key:
                # Use ParadexSubkey for L2-only authentication
                logger.info("Using ParadexSubkey for L2-only authentication")
                self._paradex = ParadexSubkey(
                    env=env,
                    l2_private_key=self.l2_private_key,
                    l2_address=self.l2_address,
                )
            else:
                # Fallback to standard Paradex (requires L1 credentials)
                logger.warning("ParadexSubkey not available, using standard Paradex")
                self._paradex = Paradex(
                    env=env,
                    l1_private_key=self.l2_private_key,  # Try as L1 key
                )

            # Create aiohttp session with connection pooling
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30,
            )

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )

            # Authenticate
            await self.authenticate()

            self._initialized = True
            logger.info(f"Paradex client initialized (env: {self.environment})")

        except ImportError as e:
            raise APIError(f"paradex-py not installed: {e}")
        except Exception as e:
            raise APIError(f"Failed to initialize client: {e}")

    async def close(self) -> None:
        """Close client and cleanup resources."""
        if self._session:
            await self._session.close()
            self._session = None

        self._initialized = False
        logger.info("Paradex client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # ==================== Authentication ====================

    async def authenticate(self) -> bool:
        """
        Authenticate using Interactive Token.

        Interactive tokens provide zero trading fees.
        Token is valid for ~24 hours.

        Returns:
            True if authentication successful.

        Raises:
            AuthenticationError: If authentication fails.
        """
        try:
            if self._paradex is None:
                raise AuthenticationError("Client not initialized")

            # Check if using ParadexSubkey (L2-only auth)
            try:
                from paradex_py import ParadexSubkey
                is_subkey = isinstance(self._paradex, ParadexSubkey)
            except ImportError:
                is_subkey = False

            if is_subkey:
                # ParadexSubkey requires onboarding() to get JWT
                logger.info("Using ParadexSubkey - calling onboarding for JWT")
                try:
                    # Call onboarding to register/authenticate with Paradex
                    result = self._paradex.onboarding()
                    logger.info(f"Onboarding result type: {type(result)}")

                    # Try to extract JWT from onboarding result
                    jwt_token = None

                    if isinstance(result, str):
                        jwt_token = result
                    elif isinstance(result, dict):
                        jwt_token = result.get('jwt_token') or result.get('token') or result.get('jwt')

                    # Check account attributes
                    if not jwt_token:
                        for attr in ['jwt_token', '_jwt_token', 'jwt', '_jwt', 'token']:
                            if hasattr(self._paradex.account, attr):
                                token = getattr(self._paradex.account, attr)
                                if token:
                                    jwt_token = token
                                    logger.info(f"Found JWT in account.{attr}")
                                    break

                    # Check paradex object itself
                    if not jwt_token:
                        for attr in ['jwt_token', '_jwt_token', 'jwt', 'token']:
                            if hasattr(self._paradex, attr):
                                token = getattr(self._paradex, attr)
                                if token:
                                    jwt_token = token
                                    logger.info(f"Found JWT in paradex.{attr}")
                                    break

                    # Check api_client headers
                    if not jwt_token and hasattr(self._paradex, 'api_client'):
                        api_client = self._paradex.api_client
                        if hasattr(api_client, 'headers'):
                            auth_header = api_client.headers.get('Authorization', '')
                            if auth_header.startswith('Bearer '):
                                jwt_token = auth_header[7:]
                                logger.info("Found JWT in api_client headers")

                    if jwt_token:
                        self.jwt_token = jwt_token
                        logger.info(f"JWT token obtained (length: {len(jwt_token)})")
                    else:
                        logger.warning("Could not extract JWT from SDK, will use SDK methods")
                        self.jwt_token = None

                    self.jwt_expires_at = int(time.time()) + 23 * 3600
                    logger.info("Authentication successful (ParadexSubkey onboarding)")
                    return True

                except Exception as e:
                    logger.error(f"ParadexSubkey onboarding failed: {e}")
                    raise AuthenticationError(f"ParadexSubkey onboarding failed: {e}")

            # For standard Paradex class, try to get JWT token
            auth_response = await self._make_auth_request()

            if auth_response and "jwt_token" in auth_response:
                self.jwt_token = auth_response["jwt_token"]
                self.jwt_expires_at = int(time.time()) + 23 * 3600
                logger.info("Authentication successful (Interactive Token)")
                return True

            raise AuthenticationError("Invalid authentication response")

        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")

    async def _make_auth_request(self) -> Dict[str, Any]:
        """
        Make authentication request to Paradex.

        Uses paradex-py SDK for Starknet signature.
        Only used for standard Paradex class (not ParadexSubkey).
        """
        if self._paradex is None:
            raise AuthenticationError("Client not initialized")

        try:
            # For standard Paradex class, try to get JWT via account
            if hasattr(self._paradex.account, 'get_jwt_token'):
                response = self._paradex.account.get_jwt_token()
                return {"jwt_token": response}
            elif hasattr(self._paradex.account, 'jwt_token'):
                return {"jwt_token": self._paradex.account.jwt_token}
            else:
                logger.info("SDK handles authentication internally")
                return {"jwt_token": "SDK_MANAGED"}

        except Exception as e:
            logger.warning(f"SDK auth failed, trying direct: {e}")
            return await self._direct_auth()

    async def _direct_auth(self) -> Dict[str, Any]:
        """Direct API authentication as fallback."""
        raise AuthenticationError("Direct auth not implemented, paradex-py required")

    async def ensure_authenticated(self) -> bool:
        """
        Ensure we have a valid authentication token.

        Refreshes token if expired or expiring soon.

        Returns:
            True if authenticated.
        """
        # Refresh 1 hour before expiry
        if time.time() >= self.jwt_expires_at - 3600:
            logger.info("Refreshing authentication token")
            return await self.authenticate()

        return bool(self.jwt_token)

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.jwt_token and self.jwt_token != "SDK_MANAGED":
            headers["Authorization"] = f"Bearer {self.jwt_token}"
        elif self._paradex is not None:
            # Try to get JWT from SDK's internal state
            jwt = self._get_sdk_jwt()
            if jwt:
                headers["Authorization"] = f"Bearer {jwt}"

        return headers

    def _get_sdk_jwt(self) -> Optional[str]:
        """Try to extract JWT from SDK's internal state."""
        if self._paradex is None:
            return None

        try:
            # Check api_client headers
            if hasattr(self._paradex, 'api_client'):
                api_client = self._paradex.api_client
                if hasattr(api_client, 'headers'):
                    auth = api_client.headers.get('Authorization', '')
                    if auth.startswith('Bearer '):
                        return auth[7:]

            # Check account jwt_token
            if hasattr(self._paradex, 'account'):
                account = self._paradex.account
                for attr in ['jwt_token', '_jwt_token', 'jwt']:
                    if hasattr(account, attr):
                        token = getattr(account, attr)
                        if token:
                            return token

            # Check paradex jwt_token
            for attr in ['jwt_token', '_jwt_token', 'jwt']:
                if hasattr(self._paradex, attr):
                    token = getattr(self._paradex, attr)
                    if token:
                        return token

        except Exception as e:
            logger.debug(f"Could not extract JWT from SDK: {e}")

        return None

    # ==================== HTTP Methods ====================

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        auth_required: bool = True,
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE).
            endpoint: API endpoint.
            params: Query parameters.
            data: Request body data.
            auth_required: Whether authentication is required.

        Returns:
            Response data.

        Raises:
            APIError: If request fails after retries.
        """
        if not self._session:
            raise APIError("Client not initialized")

        if auth_required:
            await self.ensure_authenticated()

        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()

        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                await self._rate_limit()

                async with self._session.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=data,
                ) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        return response_data

                    # Handle specific error codes
                    if response.status == 401:
                        # Token expired, refresh and retry
                        await self.authenticate()
                        continue

                    if response.status == 429:
                        # Rate limited
                        retry_after = float(response.headers.get("Retry-After", 5))
                        raise RateLimitError(
                            f"Rate limit exceeded",
                            retry_after=retry_after
                        )

                    if response.status == 400:
                        error_msg = response_data.get("message", "Bad request")
                        if "insufficient" in error_msg.lower():
                            raise InsufficientBalanceError(
                                error_msg,
                                required=0,
                                available=0
                            )
                        raise APIError(error_msg, response.status, response_data)

                    raise APIError(
                        f"Request failed: {response_data.get('message', 'Unknown error')}",
                        response.status,
                        response_data
                    )

            except RateLimitError as e:
                logger.warning(f"Rate limited, waiting {e.retry_after}s")
                await asyncio.sleep(e.retry_after)
                last_error = e

            except aiohttp.ClientError as e:
                last_error = APIError(f"Network error: {e}")
                delay = exponential_backoff(attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

            except (InsufficientBalanceError, AuthenticationError):
                raise

            except Exception as e:
                last_error = APIError(f"Unexpected error: {e}")
                delay = exponential_backoff(attempt)
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)

        raise last_error or APIError("Request failed after retries")

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        now = time.time()
        elapsed = now - self._last_request_time

        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)

        self._last_request_time = time.time()

    # ==================== Market Data Methods ====================

    async def get_bbo(self, market: str) -> Optional[BBO]:
        """
        Get best bid and offer.

        Args:
            market: Market symbol (e.g., "BTC-USD-PERP").

        Returns:
            BBO object or None if unavailable.
        """
        try:
            response = await self._request(
                "GET",
                f"/markets/{market}/bbo",
                auth_required=False
            )

            if response:
                return BBO(
                    bid=safe_float(response.get("bid")),
                    ask=safe_float(response.get("ask")),
                    bid_size=safe_float(response.get("bid_size")),
                    ask_size=safe_float(response.get("ask_size")),
                    timestamp=time.time()
                )

        except Exception as e:
            logger.error(f"Failed to get BBO: {e}")

        return None

    async def get_orderbook(self, market: str, depth: int = 5) -> Optional[Orderbook]:
        """
        Get orderbook snapshot.

        Args:
            market: Market symbol.
            depth: Number of levels to retrieve.

        Returns:
            Orderbook object or None if unavailable.
        """
        try:
            response = await self._request(
                "GET",
                f"/markets/{market}/orderbook",
                params={"depth": depth},
                auth_required=False
            )

            if response:
                bids = [
                    OrderbookLevel(price=safe_float(level[0]), size=safe_float(level[1]))
                    for level in response.get("bids", [])
                ]
                asks = [
                    OrderbookLevel(price=safe_float(level[0]), size=safe_float(level[1]))
                    for level in response.get("asks", [])
                ]

                return Orderbook(
                    bids=bids,
                    asks=asks,
                    timestamp=time.time()
                )

        except Exception as e:
            logger.error(f"Failed to get orderbook: {e}")

        return None

    async def get_recent_trades(self, market: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent trades.

        Args:
            market: Market symbol.
            limit: Maximum number of trades.

        Returns:
            List of trade dictionaries.
        """
        try:
            response = await self._request(
                "GET",
                f"/markets/{market}/trades",
                params={"limit": limit},
                auth_required=False
            )

            return response.get("results", [])

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    async def get_market_info(self, market: str) -> Optional[Dict[str, Any]]:
        """
        Get market information.

        Args:
            market: Market symbol.

        Returns:
            Market info dictionary or None.
        """
        try:
            response = await self._request(
                "GET",
                f"/markets/{market}",
                auth_required=False
            )
            return response

        except Exception as e:
            logger.error(f"Failed to get market info: {e}")
            return None

    # ==================== Account Methods ====================

    async def get_balance(self) -> Optional[float]:
        """
        Get USDC balance.

        Returns:
            Balance in USDC or None if unavailable.
        """
        try:
            response = await self._request("GET", "/account")

            if response:
                # Look for USDC balance
                balances = response.get("balances", [])
                for balance in balances:
                    if balance.get("token") == "USDC":
                        return safe_float(balance.get("size"))

                # Fallback to account equity
                return safe_float(response.get("account_value"))

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")

        return None

    async def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get full account information.

        Returns:
            AccountInfo object or None.
        """
        try:
            response = await self._request("GET", "/account")

            if response:
                positions = await self.get_positions()

                return AccountInfo(
                    balance=safe_float(response.get("account_value")),
                    available_balance=safe_float(response.get("free_collateral")),
                    unrealized_pnl=safe_float(response.get("unrealized_pnl")),
                    margin_used=safe_float(response.get("margin_used")),
                    leverage=int(response.get("leverage", 1)),
                    positions=positions
                )

        except Exception as e:
            logger.error(f"Failed to get account info: {e}")

        return None

    async def get_positions(self, market: Optional[str] = None) -> List[Position]:
        """
        Get open positions.

        Args:
            market: Filter by market (optional).

        Returns:
            List of Position objects.
        """
        try:
            params = {}
            if market:
                params["market"] = market

            response = await self._request("GET", "/positions", params=params)

            positions = []
            for pos in response.get("results", []):
                size = safe_float(pos.get("size"))
                if size == 0:
                    continue

                positions.append(Position(
                    market=pos.get("market"),
                    side=PositionSide.LONG if size > 0 else PositionSide.SHORT,
                    size=abs(size),
                    entry_price=safe_float(pos.get("avg_entry_price")),
                    unrealized_pnl=safe_float(pos.get("unrealized_pnl")),
                    leverage=int(pos.get("leverage", 1)),
                    liquidation_price=safe_float(pos.get("liquidation_price")),
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def has_open_position(self, market: str) -> bool:
        """
        Check if there's an open position for a market.

        Args:
            market: Market symbol.

        Returns:
            True if position exists.
        """
        positions = await self.get_positions(market)
        return len(positions) > 0

    # ==================== Order Methods ====================

    async def place_market_order(
        self,
        market: str,
        side: str,
        size: str,
        reduce_only: bool = False,
        client_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a market order.

        Args:
            market: Market symbol.
            side: "BUY" or "SELL".
            size: Order size.
            reduce_only: If True, only reduces position.
            client_id: Optional client order ID.

        Returns:
            Order response dictionary or None.
        """
        try:
            data = {
                "market": market,
                "side": side.upper(),
                "type": "MARKET",
                "size": str(size),
                "reduce_only": reduce_only,
            }

            if client_id:
                data["client_id"] = client_id
            else:
                data["client_id"] = generate_order_id()

            response = await self._request("POST", "/orders", data=data)

            logger.info(
                f"Market order placed: {side} {size} {market} "
                f"(id: {response.get('id')})"
            )

            return response

        except InsufficientBalanceError:
            raise
        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            raise OrderError(f"Failed to place market order: {e}")

    async def place_limit_order(
        self,
        market: str,
        side: str,
        size: str,
        price: str,
        post_only: bool = True,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Place a limit order.

        Args:
            market: Market symbol.
            side: "BUY" or "SELL".
            size: Order size.
            price: Limit price.
            post_only: If True, order only executes as maker.
            reduce_only: If True, only reduces position.
            time_in_force: Order time in force (GTC, IOC, FOK).
            client_id: Optional client order ID.

        Returns:
            Order response dictionary or None.
        """
        try:
            instruction = "POST_ONLY" if post_only else "GTC"

            data = {
                "market": market,
                "side": side.upper(),
                "type": "LIMIT",
                "size": str(size),
                "price": str(price),
                "instruction": instruction,
                "reduce_only": reduce_only,
            }

            if client_id:
                data["client_id"] = client_id
            else:
                data["client_id"] = generate_order_id()

            response = await self._request("POST", "/orders", data=data)

            logger.info(
                f"Limit order placed: {side} {size} @ {price} {market} "
                f"(id: {response.get('id')}, post_only: {post_only})"
            )

            return response

        except InsufficientBalanceError:
            raise
        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            raise OrderError(f"Failed to place limit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully.
        """
        try:
            await self._request("DELETE", f"/orders/{order_id}")
            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, market: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            market: Filter by market (optional).

        Returns:
            Number of orders cancelled.
        """
        try:
            params = {}
            if market:
                params["market"] = market

            response = await self._request("DELETE", "/orders", params=params)
            count = response.get("cancelled_count", 0)

            logger.info(f"Cancelled {count} orders" + (f" for {market}" if market else ""))
            return count

        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return 0

    async def get_open_orders(self, market: Optional[str] = None) -> List[Order]:
        """
        Get open orders.

        Args:
            market: Filter by market (optional).

        Returns:
            List of Order objects.
        """
        try:
            params = {"status": "OPEN"}
            if market:
                params["market"] = market

            response = await self._request("GET", "/orders", params=params)

            orders = []
            for order_data in response.get("results", []):
                orders.append(Order(
                    order_id=order_data.get("id"),
                    market=order_data.get("market"),
                    side=OrderSide(order_data.get("side")),
                    order_type=OrderType(order_data.get("type")),
                    size=safe_float(order_data.get("size")),
                    price=safe_float(order_data.get("price")),
                    status=OrderStatus(order_data.get("status")),
                    filled_size=safe_float(order_data.get("filled_size")),
                    avg_fill_price=safe_float(order_data.get("avg_fill_price")),
                    client_id=order_data.get("client_id"),
                ))

            return orders

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    # ==================== Position Management ====================

    async def close_position(self, market: str) -> Optional[Dict[str, Any]]:
        """
        Close position for a market.

        Places a market order to close the entire position.

        Args:
            market: Market symbol.

        Returns:
            Order response or None if no position.
        """
        try:
            positions = await self.get_positions(market)

            if not positions:
                logger.info(f"No position to close for {market}")
                return None

            position = positions[0]

            # Determine order side (opposite of position)
            side = "SELL" if position.side == PositionSide.LONG else "BUY"

            response = await self.place_market_order(
                market=market,
                side=side,
                size=format_size(position.size),
                reduce_only=True,
            )

            logger.info(f"Position closed for {market}: {position.side.value} {position.size}")
            return response

        except Exception as e:
            logger.error(f"Failed to close position for {market}: {e}")
            raise

    async def close_all_positions(self) -> int:
        """
        Close all open positions.

        Returns:
            Number of positions closed.
        """
        try:
            positions = await self.get_positions()
            closed = 0

            for position in positions:
                try:
                    await self.close_position(position.market)
                    closed += 1
                except Exception as e:
                    logger.error(f"Failed to close position for {position.market}: {e}")

            logger.info(f"Closed {closed} positions")
            return closed

        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return 0

    # ==================== Utility Methods ====================

    async def ping(self) -> bool:
        """
        Check API connectivity.

        Returns:
            True if API is reachable.
        """
        try:
            await self._request("GET", "/system/time", auth_required=False)
            return True
        except Exception:
            return False

    async def get_server_time(self) -> Optional[float]:
        """
        Get server timestamp.

        Returns:
            Server timestamp or None.
        """
        try:
            response = await self._request("GET", "/system/time", auth_required=False)
            return safe_float(response.get("server_time"))
        except Exception:
            return None

    async def measure_latency(self, samples: int = 5) -> float:
        """
        Measure API latency.

        Args:
            samples: Number of samples to average.

        Returns:
            Average latency in milliseconds.
        """
        latencies = []

        for _ in range(samples):
            start = time.time()
            await self.ping()
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            await asyncio.sleep(0.1)

        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"API latency: {avg_latency:.1f}ms (avg of {samples} samples)")

        return avg_latency
