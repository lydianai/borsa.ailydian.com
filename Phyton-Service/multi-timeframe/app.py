"""
MULTI-TIMEFRAME ANALYSIS SERVICE
===============================

Flask service for advanced multi-timeframe confluence analysis
Port: 5010

Endpoints:
- POST /mtf/data - Add timeframe data
- POST /mtf/analyze - Perform multi-timeframe analysis
- GET /mtf/opportunities - Get trading opportunities
- GET /mtf/outlook - Get market outlook
- GET /mtf/zones - Get confluence zones
- GET /mtf/patterns - Get harmonic patterns
- GET /mtf/divergences - Get divergences
- GET /health - Health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_timeframe import MultiTimeframeAnalyzer, Timeframe

app = Flask(__name__)
CORS(app)

# Initialize multi-timeframe analyzer
mta = MultiTimeframeAnalyzer(
    symbol="BTCUSDT",
    primary_timeframe=Timeframe.ONE_HOUR,
    timeframes_to_analyze=[
        Timeframe.FIFTEEN_MINUTE,
        Timeframe.ONE_HOUR,
        Timeframe.FOUR_HOUR,
        Timeframe.ONE_DAY,
        Timeframe.ONE_WEEK
    ]
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Multi-Timeframe Analysis Service',
        'version': '1.0.0',
        'port': 5010
    })

@app.route('/mtf/data', methods=['POST'])
def add_timeframe_data():
    """
    Add OHLCV data for a specific timeframe
    
    Expected JSON:
    {
        "timeframe": "1h",
        "symbol": "BTCUSDT",
        "data": [
            {
                "timestamp": "2023-01-01T00:00:00Z",
                "open": 40000.0,
                "high": 41000.0,
                "low": 39500.0,
                "close": 40500.0,
                "volume": 1000.0
            }
        ]
    }
    """
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['timeframe', 'symbol', 'data']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Map timeframe string to enum
        timeframe_map = {
            '1m': Timeframe.ONE_MINUTE,
            '5m': Timeframe.FIVE_MINUTE,
            '15m': Timeframe.FIFTEEN_MINUTE,
            '30m': Timeframe.THIRTY_MINUTE,
            '1h': Timeframe.ONE_HOUR,
            '2h': Timeframe.TWO_HOUR,
            '4h': Timeframe.FOUR_HOUR,
            '6h': Timeframe.SIX_HOUR,
            '12h': Timeframe.TWELVE_HOUR,
            '1d': Timeframe.ONE_DAY,
            '3d': Timeframe.THREE_DAY,
            '1w': Timeframe.ONE_WEEK,
            '1M': Timeframe.ONE_MONTH
        }
        
        timeframe_enum = timeframe_map.get(data['timeframe'])
        if not timeframe_enum:
            return jsonify({
                'success': False,
                'error': f'Unsupported timeframe: {data["timeframe"]}'
            }), 400
        
        # Convert data to DataFrame
        df_data = []
        for item in data['data']:
            df_data.append({
                'timestamp': pd.to_datetime(item['timestamp']),
                'open': float(item['open']),
                'high': float(item['high']),
                'low': float(item['low']),
                'close': float(item['close']),
                'volume': float(item['volume'])
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        
        # Add to analyzer
        mta.add_timeframe_data(timeframe_enum, df)
        
        return jsonify({
            'success': True,
            'message': f'Data added for {data["timeframe"]} timeframe',
            'symbol': data['symbol'],
            'timeframe': data['timeframe'],
            'candles_added': len(df)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mtf/analyze', methods=['POST'])
def perform_analysis():
    """
    Perform multi-timeframe analysis
    
    Expected JSON:
    {
        "symbol": "BTCUSDT"
    }
    """
    try:
        data = request.json or {}
        symbol = data.get('symbol', 'BTCUSDT')
        
        # Update symbol if needed
        if symbol != mta.symbol:
            mta.symbol = symbol
        
        # Perform analysis
        analysis = mta.analyze_all_timeframes()
        
        # Convert analysis to JSON-serializable format
        confluence_zones = []
        for zone in analysis.confluence_zones:
            confluence_zones.append({
                'price_level': float(zone.price_level),
                'confluence_score': float(zone.confluence_score),
                'confluence_level': zone.confluence_level.value,
                'support_resistance': zone.support_resistance,
                'harmonic_alignment': zone.harmonic_alignment,
                'divergence_present': zone.divergence_present,
                'timeframe_signals': len(zone.timeframe_signals),
                'timestamp': zone.timestamp.isoformat()
            })
        
        harmonic_patterns = []
        for pattern in analysis.harmonic_patterns:
            harmonic_patterns.append({
                'type': pattern.get('type', 'unknown'),
                'timeframe': pattern.get('timeframe', 'unknown'),
                'confidence': float(pattern.get('confidence', 0.0)),
                'timestamp': pattern.get('timestamp', datetime.now().isoformat())
            })
        
        divergences = []
        for div in analysis.divergences:
            divergences.append({
                'type': div.get('type', 'unknown'),
                'timeframe': div.get('timeframe', 'unknown'),
                'confidence': float(div.get('confidence', 0.0)),
                'timestamp': div.get('timestamp', datetime.now().isoformat())
            })
        
        key_levels = []
        for level in analysis.key_levels:
            key_levels.append({
                'price': float(level.get('price', 0.0)),
                'confluence_score': float(level.get('confluence_score', 0.0)),
                'confluence_level': level.get('confluence_level', 'unknown'),
                'support_resistance': level.get('support_resistance', 'neutral'),
                'harmonic_alignment': level.get('harmonic_alignment', False),
                'divergence_present': level.get('divergence_present', False),
                'timeframes': level.get('timeframes', []),
                'timestamp': level.get('timestamp', datetime.now().isoformat())
            })
        
        return jsonify({
            'success': True,
            'analysis': {
                'symbol': analysis.symbol,
                'primary_timeframe': analysis.primary_timeframe.value[0],
                'dominant_trend': analysis.dominant_trend.value,
                'trend_strength': float(analysis.trend_strength),
                'momentum_alignment': float(analysis.momentum_alignment),
                'confluence_zones': confluence_zones,
                'harmonic_patterns': harmonic_patterns,
                'divergences': divergences,
                'key_levels': key_levels,
                'timestamp': analysis.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mtf/opportunities', methods=['GET'])
def get_opportunities():
    """
    Get trading opportunities
    
    Query parameters:
    - limit: Number of opportunities to return (default: 10)
    """
    try:
        limit = int(request.args.get('limit', 10))
        
        # Get opportunities
        opportunities = mta.get_trading_opportunities()
        
        # Apply limit
        limited_opportunities = opportunities[:limit]
        
        # Convert to JSON-serializable format
        opportunities_json = []
        for opp in limited_opportunities:
            opp_json = {
                'type': opp['type'],
                'price': float(opp['price']),
                'direction': opp['direction'],
                'confidence': float(opp['confidence']),
                'strength': float(opp['strength']),
                'reason': opp['reason'],
                'timestamp': opp.get('timestamp', datetime.now().isoformat())
            }
            
            # Add optional fields
            for field in ['support_resistance', 'harmonic', 'divergence', 'timeframes', 
                         'pattern_type', 'divergence_type', 'timeframe']:
                if field in opp:
                    opp_json[field] = opp[field]
            
            opportunities_json.append(opp_json)
        
        return jsonify({
            'success': True,
            'opportunities': opportunities_json,
            'count': len(opportunities_json),
            'total': len(opportunities)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mtf/outlook', methods=['GET'])
def get_outlook():
    """
    Get market outlook
    """
    try:
        outlook = mta.get_market_outlook()
        
        # Convert key levels to JSON-serializable format
        key_levels = []
        for level in outlook.get('key_levels', []):
            key_levels.append({
                'price': float(level['price']),
                'type': level['type'],
                'strength': float(level['strength']),
                'harmonic': level['harmonic']
            })
        
        outlook_serializable = {
            'trend': outlook['trend'],
            'trend_strength': float(outlook['trend_strength']),
            'momentum': float(outlook['momentum']),
            'outlook': outlook['outlook'],
            'confidence': float(outlook['confidence']),
            'key_levels': key_levels,
            'opportunities': outlook['opportunities'],
            'timestamp': outlook['timestamp']
        }
        
        return jsonify({
            'success': True,
            'outlook': outlook_serializable
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mtf/zones', methods=['GET'])
def get_confluence_zones():
    """
    Get confluence zones
    
    Query parameters:
    - limit: Number of zones to return (default: 10)
    - min_score: Minimum confluence score (default: 3.0)
    """
    try:
        limit = int(request.args.get('limit', 10))
        min_score = float(request.args.get('min_score', 3.0))
        
        # Get latest analysis
        if not mta.analysis_history:
            return jsonify({
                'success': False,
                'error': 'No analysis performed yet'
            }), 400
        
        latest_analysis = mta.analysis_history[-1]
        
        # Filter zones by minimum score
        filtered_zones = [
            zone for zone in latest_analysis.confluence_zones 
            if zone.confluence_score >= min_score
        ]
        
        # Sort by confluence score
        filtered_zones.sort(key=lambda x: x.confluence_score, reverse=True)
        
        # Apply limit
        limited_zones = filtered_zones[:limit]
        
        # Convert to JSON-serializable format
        zones_json = []
        for zone in limited_zones:
            zone_json = {
                'price_level': float(zone.price_level),
                'confluence_score': float(zone.confluence_score),
                'confluence_level': zone.confluence_level.value,
                'support_resistance': zone.support_resistance,
                'harmonic_alignment': zone.harmonic_alignment,
                'divergence_present': zone.divergence_present,
                'timeframe_signals': len(zone.timeframe_signals),
                'timestamp': zone.timestamp.isoformat()
            }
            zones_json.append(zone_json)
        
        return jsonify({
            'success': True,
            'zones': zones_json,
            'count': len(zones_json),
            'total_filtered': len(filtered_zones)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mtf/patterns', methods=['GET'])
def get_harmonic_patterns():
    """
    Get harmonic patterns
    
    Query parameters:
    - limit: Number of patterns to return (default: 5)
    - min_confidence: Minimum confidence (default: 0.5)
    """
    try:
        limit = int(request.args.get('limit', 5))
        min_confidence = float(request.args.get('min_confidence', 0.5))
        
        # Get latest analysis
        if not mta.analysis_history:
            return jsonify({
                'success': False,
                'error': 'No analysis performed yet'
            }), 400
        
        latest_analysis = mta.analysis_history[-1]
        
        # Filter patterns by minimum confidence
        filtered_patterns = [
            pattern for pattern in latest_analysis.harmonic_patterns 
            if pattern.get('confidence', 0.0) >= min_confidence
        ]
        
        # Sort by confidence
        filtered_patterns.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Apply limit
        limited_patterns = filtered_patterns[:limit]
        
        # Convert to JSON-serializable format
        patterns_json = []
        for pattern in limited_patterns:
            pattern_json = {
                'type': pattern.get('type', 'unknown'),
                'timeframe': pattern.get('timeframe', 'unknown'),
                'confidence': float(pattern.get('confidence', 0.0)),
                'timestamp': pattern.get('timestamp', datetime.now().isoformat())
            }
            patterns_json.append(pattern_json)
        
        return jsonify({
            'success': True,
            'patterns': patterns_json,
            'count': len(patterns_json),
            'total_filtered': len(filtered_patterns)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/mtf/divergences', methods=['GET'])
def get_divergences():
    """
    Get divergences
    
    Query parameters:
    - limit: Number of divergences to return (default: 5)
    - min_confidence: Minimum confidence (default: 0.5)
    """
    try:
        limit = int(request.args.get('limit', 5))
        min_confidence = float(request.args.get('min_confidence', 0.5))
        
        # Get latest analysis
        if not mta.analysis_history:
            return jsonify({
                'success': False,
                'error': 'No analysis performed yet'
            }), 400
        
        latest_analysis = mta.analysis_history[-1]
        
        # Filter divergences by minimum confidence
        filtered_divergences = [
            div for div in latest_analysis.divergences 
            if div.get('confidence', 0.0) >= min_confidence
        ]
        
        # Sort by confidence
        filtered_divergences.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
        
        # Apply limit
        limited_divergences = filtered_divergences[:limit]
        
        # Convert to JSON-serializable format
        divergences_json = []
        for div in limited_divergences:
            div_json = {
                'type': div.get('type', 'unknown'),
                'timeframe': div.get('timeframe', 'unknown'),
                'confidence': float(div.get('confidence', 0.0)),
                'timestamp': div.get('timestamp', datetime.now().isoformat())
            }
            divergences_json.append(div_json)
        
        return jsonify({
            'success': True,
            'divergences': divergences_json,
            'count': len(divergences_json),
            'total_filtered': len(filtered_divergences)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 MULTI-TIMEFRAME ANALYSIS SERVICE")
    print("="*60)
    print("🎯 Advanced Multi-Timeframe Confluence Analysis")
    print("📊 Timeframe Harmony Scoring & Confluence Detection")
    print("📈 Harmonic Patterns, Divergences, Trend Alignment")
    print("🌐 Server: http://localhost:5010")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5010, debug=True)