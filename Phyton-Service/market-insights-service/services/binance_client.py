"""
BINANCE FUTURES API CLIENT
Fetches funding rate, open interest, and market data
White-hat compliant: Uses public API endpoints only
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Binance Futures API Base URL
BINANCE_FUTURES_BASE_URL = 'https://fapi.binance.com'


def fetch_funding_rate(symbol: str = 'BTCUSDT', limit: int = 100) -> Dict:
    """
    Fetch funding rate history from Binance Futures API

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        limit: Number of records to fetch (max 1000)

    Returns:
        Dict with success status and data
    """
    try:
        url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/fundingRate"
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)  # Max 1000 from Binance
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Format data for our application
        formatted_data = []
        for item in data:
            formatted_data.append({
                'symbol': item.get('symbol'),
                'funding_rate': float(item.get('fundingRate', 0)),
                'funding_time': item.get('fundingTime'),
                'timestamp': datetime.fromtimestamp(item.get('fundingTime', 0) / 1000).isoformat()
            })

        logger.info(f"[Binance Client] Fetched {len(formatted_data)} funding rate records for {symbol}")
        return {
            'success': True,
            'data': formatted_data,
            'source': 'binance'
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[Binance Client] Error fetching funding rate: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'source': 'binance'
        }


def fetch_open_interest(symbol: str = 'BTCUSDT') -> Dict:
    """
    Fetch current open interest from Binance Futures API

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')

    Returns:
        Dict with success status and data
    """
    try:
        url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/openInterest"
        params = {
            'symbol': symbol.upper()
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        formatted_data = {
            'symbol': data.get('symbol'),
            'open_interest': float(data.get('openInterest', 0)),
            'timestamp': datetime.fromtimestamp(data.get('time', 0) / 1000).isoformat(),
            'time_ms': data.get('time')
        }

        logger.info(f"[Binance Client] Fetched open interest for {symbol}: {formatted_data['open_interest']}")
        return {
            'success': True,
            'data': formatted_data,
            'source': 'binance'
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[Binance Client] Error fetching open interest: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'source': 'binance'
        }


def fetch_premium_index(symbol: str = 'BTCUSDT') -> Dict:
    """
    Fetch premium index and current funding rate from Binance

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')

    Returns:
        Dict with premium index, current funding rate, next funding time
    """
    try:
        url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/premiumIndex"
        params = {
            'symbol': symbol.upper()
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        formatted_data = {
            'symbol': data.get('symbol'),
            'mark_price': float(data.get('markPrice', 0)),
            'index_price': float(data.get('indexPrice', 0)),
            'last_funding_rate': float(data.get('lastFundingRate', 0)),
            'next_funding_time': data.get('nextFundingTime'),
            'next_funding_timestamp': datetime.fromtimestamp(data.get('nextFundingTime', 0) / 1000).isoformat(),
            'time_ms': data.get('time')
        }

        logger.info(f"[Binance Client] Fetched premium index for {symbol}")
        return {
            'success': True,
            'data': formatted_data,
            'source': 'binance'
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[Binance Client] Error fetching premium index: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'source': 'binance'
        }


def fetch_24h_ticker(symbol: str = 'BTCUSDT') -> Dict:
    """
    Fetch 24h ticker price statistics

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')

    Returns:
        Dict with price, volume, and 24h statistics
    """
    try:
        url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/ticker/24hr"
        params = {
            'symbol': symbol.upper()
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        formatted_data = {
            'symbol': data.get('symbol'),
            'price': float(data.get('lastPrice', 0)),
            'price_change': float(data.get('priceChange', 0)),
            'price_change_percent': float(data.get('priceChangePercent', 0)),
            'high_price': float(data.get('highPrice', 0)),
            'low_price': float(data.get('lowPrice', 0)),
            'volume': float(data.get('volume', 0)),
            'quote_volume': float(data.get('quoteVolume', 0)),
            'open_time': data.get('openTime'),
            'close_time': data.get('closeTime')
        }

        logger.info(f"[Binance Client] Fetched 24h ticker for {symbol}: ${formatted_data['price']}")
        return {
            'success': True,
            'data': formatted_data,
            'source': 'binance'
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[Binance Client] Error fetching 24h ticker: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'source': 'binance'
        }


def test_connection() -> Dict:
    """
    Test Binance API connection

    Returns:
        Dict with success status
    """
    try:
        url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/ping"
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        logger.info("[Binance Client] Connection test successful")
        return {
            'success': True,
            'message': 'Binance API connection successful',
            'source': 'binance'
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"[Binance Client] Connection test failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'source': 'binance'
        }
