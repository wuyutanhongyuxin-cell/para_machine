"""
Setup script for Paradex Trader.

Install with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="paradex_trader",
    version="1.0.0",
    description="Self-learning cryptocurrency trading bot for Paradex DEX",
    author="Paradex Trader Team",
    python_requires=">=3.9",
    packages=["paradex_trader"],
    package_dir={"paradex_trader": "."},
    install_requires=[
        "paradex-py>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "aiosqlite>=0.19.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "river>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "paradex-trader=paradex_trader.main:main",
        ],
    },
)
