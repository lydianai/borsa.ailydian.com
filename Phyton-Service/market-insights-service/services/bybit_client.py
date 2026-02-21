"""
BYBIT API CLIENT (FALLBACK)
Fallback when Binance fails
"""

import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
BYBIT_BASE_URL = 'https://api.bybit.com'


def fetch_funding_rate_fallback(symbol: str = 'BTCUSDT', limit: int = 100):
    """Fallback funding rate from Bybit"""
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/funding/history"
        params = {'category': 'linear', 'symbol': symbol.upper(), 'limit': limit}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('retCode') == 0:
            formatted_data = []
            for item in data.get('result', {}).get('list', []):
                formatted_data.append({
                    'symbol': item.get('symbol'),
                    'funding_rate': float(item.get('fundingRate', 0)),
                    'funding_time': int(item.get('fundingRateTimestamp', 0)),
                    'timestamp': datetime.fromtimestamp(int(item.get('fundingRateTimestamp', 0)) / 1000).isoformat()
                })

            logger.info(f"[Bybit Fallback] Fetched {len(formatted_data)} funding rates")
            return {'success': True, 'data': formatted_data, 'source': 'bybit'}

        return {'success': False, 'error': 'Bybit API error', 'source': 'bybit'}

    except Exception as e:
        logger.error(f"[Bybit Fallback] Error: {str(e)}")
        return {'success': False, 'error': str(e), 'source': 'bybit'}


def fetch_ticker_fallback(symbol: str = 'BTCUSDT'):
    """Fallback ticker from Bybit"""
    try:
        url = f"{BYBIT_BASE_URL}/v5/market/tickers"
        params = {'category': 'linear', 'symbol': symbol.upper()}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('retCode') == 0:
            ticker = data.get('result', {}).get('list', [{}])[0]
            formatted_data = {
                'symbol': ticker.get('symbol'),
                'price': float(ticker.get('lastPrice', 0)),
                'high_price': float(ticker.get('highPrice24h', 0)),
                'low_price': float(ticker.get('lowPrice24h', 0)),
                'volume': float(ticker.get('volume24h', 0)),
            }

            logger.info(f"[Bybit Fallback] Fetched ticker: ${formatted_data['price']}")
            return {'success': True, 'data': formatted_data, 'source': 'bybit'}

        return {'success': False, 'error': 'Bybit API error', 'source': 'bybit'}

    except Exception as e:
        logger.error(f"[Bybit Fallback] Error: {str(e)}")
        return {'success': False, 'error': str(e), 'source': 'bybit'}
