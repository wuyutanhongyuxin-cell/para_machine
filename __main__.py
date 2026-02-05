"""
Entry point for running Paradex Trader as a module.

Usage:
    python -m paradex_trader [--dry-run] [--debug]
"""

import asyncio
from paradex_trader.main import main

if __name__ == "__main__":
    asyncio.run(main())
