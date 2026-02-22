"""
🔗 UNIFIED BINANCE API CLIENT
=============================
Centralized Binance API client for all services

Features:
- Request rate limiting
- Automatic retry with exponential backoff
- Error handling
- Response caching
- WebSocket support
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import hmac


class BinanceClient:
    """Unified Binance API client"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        cache = None
    ):
        """
        Initialize Binance client

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            testnet: Use testnet instead of production
            cache: RedisCache instance for caching
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.cache = cache

        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
        else:
            self.base_url = "https://api.binance.com/api"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AiLydian-Trading-Scanner/1.0'
        })

        if self.api_key:
            self.session.headers.update({
                'X-MBX-APIKEY': self.api_key
            })

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests

    def _wait_for_rate_limit(self):
        """Enforce rate limiting"""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def _generate_signature(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
        cache_ttl: Optional[int] = None
    ) -> Any:
        """
        Make API request with retry logic

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request
            cache_ttl: Cache TTL in seconds (None = no cache)

        Returns:
            Response data
        """
        params = params or {}

        # Check cache first (for GET requests)
        if method == "GET" and cache_ttl and self.cache:
            cache_key = f"{endpoint}:{str(params)}"
            cached = self.cache.get("binance", cache_key)
            if cached:
                return cached

        # Rate limiting
        self._wait_for_rate_limit()

        # Add signature if required
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)

        # Make request with retry
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                # Cache successful response
                if method == "GET" and cache_ttl and self.cache:
                    cache_key = f"{endpoint}:{str(params)}"
                    self.cache.set("binance", cache_key, data, cache_ttl)

                return data

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Binance API error after {max_retries} retries: {e}")

                print(f"⚠️  Retry {attempt + 1}/{max_retries} after error: {e}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

    # ===== Public Market Data =====

    def get_ticker_price(self, symbol: Optional[str] = None) -> Any:
        """Get latest price for a symbol or all symbols"""
        params = {'symbol': symbol} if symbol else {}
        return self._request("GET", "/v3/ticker/price", params, cache_ttl=10)

    def get_ticker_24h(self, symbol: Optional[str] = None) -> Any:
        """Get 24h ticker data"""
        params = {'symbol': symbol} if symbol else {}
        return self._request("GET", "/v3/ticker/24hr", params, cache_ttl=30)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List:
        """
        Get candlestick data

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., "1m", "5m", "1h", "1d")
            limit: Number of klines (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time

        return self._request("GET", "/v3/klines", params, cache_ttl=5)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth"""
        params = {'symbol': symbol, 'limit': limit}
        return self._request("GET", "/v3/depth", params, cache_ttl=5)

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List:
        """Get recent trades"""
        params = {'symbol': symbol, 'limit': limit}
        return self._request("GET", "/v3/trades", params, cache_ttl=5)

    def get_agg_trades(self, symbol: str, limit: int = 500) -> List:
        """Get aggregated trades"""
        params = {'symbol': symbol, 'limit': limit}
        return self._request("GET", "/v3/aggTrades", params, cache_ttl=5)

    # ===== Futures Market Data =====

    def get_futures_ticker(self, symbol: Optional[str] = None) -> Any:
        """Get futures 24h ticker"""
        params = {'symbol': symbol} if symbol else {}
        return self._request("GET", "/v1/ticker/24hr", params, cache_ttl=30)

    def get_funding_rate(self, symbol: str, limit: int = 100) -> List:
        """Get funding rate history"""
        params = {'symbol': symbol, 'limit': limit}
        return self._request("GET", "/v1/fundingRate", params, cache_ttl=60)

    def get_open_interest(self, symbol: str) -> Dict:
        """Get open interest"""
        params = {'symbol': symbol}
        return self._request("GET", "/v1/openInterest", params, cache_ttl=30)

    def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "5m",
        limit: int = 30
    ) -> List:
        """Get long/short ratio"""
        params = {'symbol': symbol, 'period': period, 'limit': limit}
        return self._request("GET", "/v1/globalLongShortAccountRatio", params, cache_ttl=60)


# Example usage
if __name__ == "__main__":
    # Create client
    client = BinanceClient()

    print("\n🔍 Testing Binance client...")

    # Test spot price
    btc_price = client.get_ticker_price("BTCUSDT")
    print(f"BTC Price: ${float(btc_price['price']):.2f}")

    # Test 24h ticker
    btc_ticker = client.get_ticker_24h("BTCUSDT")
    print(f"BTC 24h Change: {btc_ticker['priceChangePercent']}%")

    # Test klines
    klines = client.get_klines("BTCUSDT", "1h", limit=5)
    print(f"Latest 5 hourly candles fetched: {len(klines)}")

    print("\n✅ Binance client test completed")
