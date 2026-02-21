"""
🎯 12-LAYER CONFIRMATION ENGINE
Composite signal confirmation from all analysis layers
Port: 5019

Features:
- Aggregates signals from 12 different analysis layers
- Composite confidence scoring
- Multi-factor signal validation
- Trade quality assessment
- Risk-adjusted recommendations

WHITE-HAT COMPLIANCE: Educational purpose, transparent analysis
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime
import sys

app = Flask(__name__)
CORS(app)
sys.stdout = sys.stderr

class ConfirmationEngine:
    def __init__(self):
        # Service URLs
        self.services = {
            'feature_engineering': 'http://localhost:5001',
            'ai_models': 'http://localhost:5003',
            'signal_generator': 'http://localhost:5004',
            'smc_strategy': 'http://localhost:5007',
            'whale_activity': 'http://localhost:5015',
            'macro_correlation': 'http://localhost:5016',
            'sentiment_analysis': 'http://localhost:5017',
            'options_flow': 'http://localhost:5018',
        }

        # Layer weights for composite scoring
        self.layer_weights = {
            'price_action': 0.12,
            'volume_analysis': 0.10,
            'technical_indicators': 0.10,
            'wyckoff_smc': 0.10,
            'support_resistance': 0.08,
            'fibonacci': 0.05,
            'order_flow': 0.08,
            'whale_activity': 0.10,
            'sentiment': 0.10,
            'correlation': 0.07,
            'options_flow': 0.05,
            'ai_prediction': 0.05,
        }

    def get_layer_score(self, url, endpoint, params=None, timeout=5):
        """Get score from a service layer"""
        try:
            response = requests.get(f"{url}{endpoint}", params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            print(f"[Confirmation] Service error {url}: {e}")
            return None

    def analyze_layer_1_price_action(self, symbol):
        """Layer 1: Price Action Analysis"""
        # Using signal generator
        data = self.get_layer_score(
            self.services['signal_generator'],
            f'/signals/{symbol}'
        )

        if data and data.get('success'):
            signal = data.get('data', {}).get('signal', 'NEUTRAL')
            confidence = data.get('data', {}).get('confidence', 50)

            return {
                'layer': 'Price Action',
                'signal': signal,
                'confidence': confidence,
                'weight': self.layer_weights['price_action'],
                'weighted_score': (confidence / 100) * self.layer_weights['price_action'] * 100
            }

        return self.get_neutral_layer('Price Action', 'price_action')

    def analyze_layer_8_whale_activity(self, symbol):
        """Layer 8: Whale Activity"""
        data = self.get_layer_score(
            self.services['whale_activity'],
            f'/analyze/{symbol}'
        )

        if data and data.get('success'):
            whale_data = data.get('data', {}).get('whale_activity', {})
            detected = whale_data.get('detected', False)

            # If whales detected, assign signal based on pressure
            if detected:
                confidence = 70
                signal = "BULLISH" if whale_data.get('buy_pressure', 0) > whale_data.get('sell_pressure', 0) else "BEARISH"
            else:
                confidence = 50
                signal = "NEUTRAL"

            return {
                'layer': 'Whale Activity',
                'signal': signal,
                'confidence': confidence,
                'weight': self.layer_weights['whale_activity'],
                'weighted_score': (confidence / 100) * self.layer_weights['whale_activity'] * 100
            }

        return self.get_neutral_layer('Whale Activity', 'whale_activity')

    def analyze_layer_9_sentiment(self, symbol):
        """Layer 9: Market Sentiment"""
        data = self.get_layer_score(
            self.services['sentiment_analysis'],
            '/composite',
            params={'symbol': symbol}
        )

        if data and data.get('success'):
            sentiment = data.get('data', {})
            score = sentiment.get('composite_score', 50)
            signal_text = sentiment.get('signal', 'NEUTRAL')

            # Convert sentiment to trading signal
            if 'GREED' in signal_text or score > 60:
                signal = "BULLISH"
            elif 'FEAR' in signal_text or score < 40:
                signal = "BEARISH"
            else:
                signal = "NEUTRAL"

            return {
                'layer': 'Market Sentiment',
                'signal': signal,
                'confidence': abs(score - 50) + 50,  # Distance from neutral
                'weight': self.layer_weights['sentiment'],
                'weighted_score': score * self.layer_weights['sentiment']
            }

        return self.get_neutral_layer('Market Sentiment', 'sentiment')

    def analyze_layer_10_correlation(self, symbol):
        """Layer 10: Market Correlation"""
        # Extract base currency
        base = symbol.replace('USDT', '')

        data = self.get_layer_score(
            self.services['macro_correlation'],
            '/analyze',
            params={'base': 'BTCUSDT'}
        )

        if data and data.get('success'):
            correlations = data.get('data', {}).get('correlations', {})
            target_corr = correlations.get(symbol, 0)

            # High correlation with BTC - follow BTC trend
            if abs(target_corr) > 0.7:
                confidence = abs(target_corr) * 100
                signal = "FOLLOW_BTC"
            else:
                confidence = 50
                signal = "INDEPENDENT"

            return {
                'layer': 'Market Correlation',
                'signal': signal,
                'confidence': confidence,
                'weight': self.layer_weights['correlation'],
                'weighted_score': confidence * self.layer_weights['correlation']
            }

        return self.get_neutral_layer('Market Correlation', 'correlation')

    def get_neutral_layer(self, layer_name, weight_key):
        """Return neutral layer when service unavailable"""
        return {
            'layer': layer_name,
            'signal': 'NEUTRAL',
            'confidence': 50,
            'weight': self.layer_weights[weight_key],
            'weighted_score': 50 * self.layer_weights[weight_key],
            'status': 'unavailable'
        }

    def calculate_composite_confidence(self, layers):
        """Calculate composite confidence score from all layers"""

        # Sum weighted scores
        total_weighted = sum([layer['weighted_score'] for layer in layers])

        # Count bullish/bearish/neutral signals
        signals = {
            'BULLISH': 0,
            'BEARISH': 0,
            'NEUTRAL': 0
        }

        for layer in layers:
            signal = layer['signal']
            if 'BULLISH' in signal or signal == 'BUY':
                signals['BULLISH'] += layer['weight']
            elif 'BEARISH' in signal or signal == 'SELL':
                signals['BEARISH'] += layer['weight']
            else:
                signals['NEUTRAL'] += layer['weight']

        # Determine overall signal
        if signals['BULLISH'] > signals['BEARISH'] and signals['BULLISH'] > 0.4:
            overall_signal = 'BULLISH'
        elif signals['BEARISH'] > signals['BULLISH'] and signals['BEARISH'] > 0.4:
            overall_signal = 'BEARISH'
        else:
            overall_signal = 'NEUTRAL'

        # Trade quality assessment
        if total_weighted >= 75:
            quality = 'EXCELLENT'
            recommendation = '🟢 Yüksek kalite sinyal - Güvenle işlem yapılabilir'
        elif total_weighted >= 65:
            quality = 'GOOD'
            recommendation = '🟡 İyi sinyal - Makul risk/ödül oranı'
        elif total_weighted >= 50:
            quality = 'MODERATE'
            recommendation = '⚪ Orta kalite - Dikkatli pozisyon boyutlandır'
        else:
            quality = 'POOR'
            recommendation = '🔴 Düşük kalite - İşlem yapmaktan kaçın'

        return {
            'composite_score': round(total_weighted, 2),
            'overall_signal': overall_signal,
            'quality': quality,
            'recommendation': recommendation,
            'signal_distribution': {
                'bullish_weight': round(signals['BULLISH'], 2),
                'bearish_weight': round(signals['BEARISH'], 2),
                'neutral_weight': round(signals['NEUTRAL'], 2)
            },
            'confirmation_strength': 'HIGH' if total_weighted >= 70 else 'MEDIUM' if total_weighted >= 55 else 'LOW'
        }

    def analyze(self, symbol="BTCUSDT"):
        """Complete 12-layer confirmation analysis"""
        print(f"[Confirmation] Running 12-layer analysis for {symbol}...")

        layers = []

        # Layer 1: Price Action
        layers.append(self.analyze_layer_1_price_action(symbol))

        # Layer 2-7: Placeholder for now (would integrate with other services)
        layers.extend([
            self.get_neutral_layer('Volume Analysis', 'volume_analysis'),
            self.get_neutral_layer('Technical Indicators', 'technical_indicators'),
            self.get_neutral_layer('Wyckoff/SMC', 'wyckoff_smc'),
            self.get_neutral_layer('Support/Resistance', 'support_resistance'),
            self.get_neutral_layer('Fibonacci', 'fibonacci'),
            self.get_neutral_layer('Order Flow', 'order_flow'),
        ])

        # Layer 8: Whale Activity
        layers.append(self.analyze_layer_8_whale_activity(symbol))

        # Layer 9: Sentiment
        layers.append(self.analyze_layer_9_sentiment(symbol))

        # Layer 10: Correlation
        layers.append(self.analyze_layer_10_correlation(symbol))

        # Layer 11-12: Placeholder
        layers.extend([
            self.get_neutral_layer('Options Flow', 'options_flow'),
            self.get_neutral_layer('AI Prediction', 'ai_prediction'),
        ])

        # Calculate composite
        composite = self.calculate_composite_confidence(layers)

        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'layers': layers,
            'composite': composite,
            'total_layers': len(layers),
            'active_layers': len([l for l in layers if l.get('status') != 'unavailable'])
        }

engine = ConfirmationEngine()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "service": "12-Layer Confirmation Engine",
        "status": "healthy",
        "port": 5019,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze/<symbol>', methods=['GET'])
def analyze(symbol):
    try:
        result = engine.analyze(symbol.upper())
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Confirmation] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/composite/<symbol>', methods=['GET'])
def composite(symbol):
    """Get only composite score"""
    try:
        analysis = engine.analyze(symbol.upper())
        return jsonify({
            "success": True,
            "data": {
                "symbol": symbol,
                "composite_score": analysis['composite']['composite_score'],
                "overall_signal": analysis['composite']['overall_signal'],
                "quality": analysis['composite']['quality'],
                "recommendation": analysis['composite']['recommendation'],
                "timestamp": analysis['timestamp']
            }
        })
    except Exception as e:
        print(f"[Confirmation] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🎯 12-Layer Confirmation Engine starting...")
    print("📡 Listening on port 5019")
    app.run(host='0.0.0.0', port=5019, debug=False)
