"""Custom exceptions for Paradex Trader."""


class ParadexTraderError(Exception):
    """Base exception for all Paradex Trader errors."""

    pass


class ConfigError(ParadexTraderError):
    """Configuration related errors."""

    pass


class APIError(ParadexTraderError):
    """API communication errors."""

    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(APIError):
    """Authentication failed."""

    pass


class RateLimitError(APIError):
    """Rate limit exceeded."""

    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


class DatabaseError(ParadexTraderError):
    """Database operation errors."""

    pass


class StrategyError(ParadexTraderError):
    """Strategy execution errors."""

    pass


class RiskLimitExceeded(ParadexTraderError):
    """Risk limit has been exceeded."""

    def __init__(self, message: str, limit_type: str, current_value: float, limit_value: float):
        super().__init__(message)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class InsufficientBalanceError(ParadexTraderError):
    """Insufficient balance for operation."""

    def __init__(self, message: str, required: float, available: float):
        super().__init__(message)
        self.required = required
        self.available = available


class OrderError(ParadexTraderError):
    """Order placement or management errors."""

    def __init__(self, message: str, order_id: str = None, reason: str = None):
        super().__init__(message)
        self.order_id = order_id
        self.reason = reason


class PositionError(ParadexTraderError):
    """Position management errors."""

    pass


class DataError(ParadexTraderError):
    """Data processing or validation errors."""

    pass
