"""
MARKET INSIGHTS MICROSERVICE
Port 5031
Provides liquidation heatmap, funding rate, open interest data
White-hat compliant
"""

import os
import sys
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

# Import services
from services import binance_client, bybit_client, liquidation_calculator
from cache import memory_cache

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market-insights.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'market-insights-service',
        'port': 5031,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/liquidation-heatmap/<symbol>', methods=['GET'])
def get_liquidation_heatmap(symbol):
    """Get liquidation heatmap data"""
    try:
        cache_key = f"heatmap_{symbol}"
        cached = memory_cache.get_cache(cache_key)
        if cached:
            logger.info(f"[Liquidation Heatmap] Cache hit for {symbol}")
            return jsonify(cached)

        # Get current price and open interest
        ticker_data = binance_client.fetch_24h_ticker(symbol)
        if not ticker_data['success']:
            # Fallback to Bybit
            ticker_data = bybit_client.fetch_ticker_fallback(symbol)

        if not ticker_data['success']:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch price data'
            }), 500

        current_price = ticker_data['data']['price']

        # Get open interest
        oi_data = binance_client.fetch_open_interest(symbol)
        open_interest = oi_data['data']['open_interest'] if oi_data['success'] else 100000

        # Calculate heatmap
        heatmap_result = liquidation_calculator.calculate_liquidation_heatmap(
            symbol, current_price, open_interest
        )

        if heatmap_result['success']:
            memory_cache.set_cache(cache_key, heatmap_result)

        return jsonify(heatmap_result)

    except Exception as e:
        logger.error(f"[Liquidation Heatmap] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/funding-rate/<symbol>', methods=['GET'])
def get_funding_rate(symbol):
    """Get funding rate history"""
    try:
        limit = request.args.get('limit', 100, type=int)
        cache_key = f"funding_{symbol}_{limit}"

        cached = memory_cache.get_cache(cache_key)
        if cached:
            return jsonify(cached)

        # Try Binance
        result = binance_client.fetch_funding_rate(symbol, limit)

        # Fallback to Bybit if Binance fails
        if not result['success']:
            logger.warning(f"[Funding Rate] Binance failed, trying Bybit fallback")
            result = bybit_client.fetch_funding_rate_fallback(symbol, limit)

        if result['success']:
            memory_cache.set_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"[Funding Rate] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/open-interest/<symbol>', methods=['GET'])
def get_open_interest(symbol):
    """Get open interest data"""
    try:
        cache_key = f"oi_{symbol}"

        cached = memory_cache.get_cache(cache_key)
        if cached:
            return jsonify(cached)

        result = binance_client.fetch_open_interest(symbol)

        if result['success']:
            memory_cache.set_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"[Open Interest] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/premium-index/<symbol>', methods=['GET'])
def get_premium_index(symbol):
    """Get premium index and current funding rate"""
    try:
        cache_key = f"premium_{symbol}"

        cached = memory_cache.get_cache(cache_key)
        if cached:
            return jsonify(cached)

        result = binance_client.fetch_premium_index(symbol)

        if result['success']:
            memory_cache.set_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"[Premium Index] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/long-short-ratio/<symbol>', methods=['GET'])
def get_long_short_ratio(symbol):
    """Get long/short ratio estimation"""
    try:
        cache_key = f"longhort_{symbol}"

        cached = memory_cache.get_cache(cache_key)
        if cached:
            return jsonify(cached)

        # Get open interest for estimation
        oi_data = binance_client.fetch_open_interest(symbol)
        open_interest = oi_data['data']['open_interest'] if oi_data['success'] else 100000

        result = liquidation_calculator.calculate_long_short_ratio(symbol, open_interest)

        if result['success']:
            memory_cache.set_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"[Long/Short Ratio] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/market-data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Get comprehensive market data (price, volume, 24h stats)"""
    try:
        cache_key = f"market_{symbol}"

        cached = memory_cache.get_cache(cache_key)
        if cached:
            return jsonify(cached)

        result = binance_client.fetch_24h_ticker(symbol)

        # Fallback to Bybit
        if not result['success']:
            result = bybit_client.fetch_ticker_fallback(symbol)

        if result['success']:
            memory_cache.set_cache(cache_key, result)

        return jsonify(result)

    except Exception as e:
        logger.error(f"[Market Data] Error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear all cache (admin endpoint)"""
    try:
        memory_cache.clear_cache()
        logger.info("[Cache] All cache cleared")
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5031))
    logger.info(f"🚀 Market Insights Service starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
