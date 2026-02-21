"""
🐋 WHALE ACTIVITY TRACKER
Real-time whale movement detection and analysis
Port: 5015

Features:
- Large volume trade detection
- Whale wallet tracking
- Abnormal volume spike detection
- Whale action signals
- Buy/Sell pressure analysis

WHITE-HAT COMPLIANCE: Educational purpose, transparent analysis
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime
from statistics import mean, stdev
import sys

app = Flask(__name__)
CORS(app)
sys.stdout = sys.stderr

class WhaleTracker:
    def __init__(self):
        self.binance_api = "https://api.binance.com"
        self.whale_threshold = 100000  # $100K minimum for whale trade

    def get_recent_trades(self, symbol, limit=500):
        """Get recent aggregated trades"""
        try:
            url = f"{self.binance_api}/api/v3/aggTrades"
            params = {"symbol": symbol, "limit": limit}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return data
        except Exception as e:
            print(f"[Whale] Recent trades error for {symbol}: {e}")
            return []

    def get_current_price(self, symbol):
        """Get current market price"""
        try:
            url = f"{self.binance_api}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data.get('price', 0))
        except Exception as e:
            print(f"[Whale] Price error for {symbol}: {e}")
            return 0

    def analyze_trades(self, trades, current_price):
        """Analyze trades for whale activity"""
        if not trades or current_price == 0:
            return {
                "whale_trades": [],
                "total_whale_volume": 0,
                "whale_buy_volume": 0,
                "whale_sell_volume": 0,
                "avg_trade_size": 0,
                "volume_spike": False
            }

        whale_trades = []
        total_volumes = []
        whale_buy_volume = 0
        whale_sell_volume = 0
        total_whale_volume = 0

        for trade in trades:
            quantity = float(trade.get('q', 0))
            price = float(trade.get('p', 0))
            is_buyer_maker = trade.get('m', False)
            timestamp = trade.get('T', 0)

            trade_value = quantity * price
            total_volumes.append(trade_value)

            # Whale trade detection (>$100K)
            if trade_value >= self.whale_threshold:
                whale_trades.append({
                    "quantity": round(quantity, 4),
                    "price": round(price, 2),
                    "value": round(trade_value, 2),
                    "side": "SELL" if is_buyer_maker else "BUY",
                    "timestamp": timestamp,
                    "time_ago": self._time_ago(timestamp)
                })

                total_whale_volume += trade_value
                if is_buyer_maker:
                    whale_sell_volume += trade_value
                else:
                    whale_buy_volume += trade_value

        # Calculate average trade size
        avg_trade_size = mean(total_volumes) if total_volumes else 0

        # Detect volume spike (whale trades > 3x average)
        volume_spike = False
        if len(whale_trades) > 0 and avg_trade_size > 0:
            whale_avg = total_whale_volume / len(whale_trades) if len(whale_trades) > 0 else 0
            volume_spike = whale_avg > (avg_trade_size * 3)

        return {
            "whale_trades": sorted(whale_trades, key=lambda x: x['value'], reverse=True)[:10],
            "total_whale_volume": round(total_whale_volume, 2),
            "whale_buy_volume": round(whale_buy_volume, 2),
            "whale_sell_volume": round(whale_sell_volume, 2),
            "avg_trade_size": round(avg_trade_size, 2),
            "volume_spike": volume_spike,
            "whale_count": len(whale_trades)
        }

    def _time_ago(self, timestamp):
        """Calculate time ago from timestamp"""
        try:
            now = datetime.now().timestamp() * 1000
            diff_ms = now - timestamp
            diff_min = diff_ms / 1000 / 60

            if diff_min < 1:
                return "Az önce"
            elif diff_min < 60:
                return f"{int(diff_min)} dakika önce"
            else:
                hours = int(diff_min / 60)
                return f"{hours} saat önce"
        except:
            return "Bilinmiyor"

    def calculate_pressure(self, whale_buy_volume, whale_sell_volume):
        """Calculate buy/sell pressure"""
        total = whale_buy_volume + whale_sell_volume
        if total == 0:
            return {
                "status": "NEUTRAL",
                "buy_pressure": 0,
                "sell_pressure": 0,
                "signal": "Whale aktivitesi tespit edilmedi"
            }

        buy_pressure = (whale_buy_volume / total) * 100
        sell_pressure = (whale_sell_volume / total) * 100

        if buy_pressure > 70:
            status = "HEAVY_BUYING"
            signal = "🐋 Whale'ler ALIM yapıyor - Güçlü yükseliş baskısı"
        elif buy_pressure > 55:
            status = "MODERATE_BUYING"
            signal = "🐋 Whale'ler alım yönünde - Pozitif momentum"
        elif sell_pressure > 70:
            status = "HEAVY_SELLING"
            signal = "⚠️ Whale'ler SATIM yapıyor - Güçlü düşüş baskısı"
        elif sell_pressure > 55:
            status = "MODERATE_SELLING"
            signal = "⚠️ Whale'ler satım yönünde - Negatif momentum"
        else:
            status = "BALANCED"
            signal = "Whale alım-satım dengeli"

        return {
            "status": status,
            "buy_pressure": round(buy_pressure, 2),
            "sell_pressure": round(sell_pressure, 2),
            "signal": signal
        }

    def detect_accumulation(self, whale_trades):
        """Detect accumulation/distribution patterns"""
        if len(whale_trades) < 3:
            return {
                "pattern": "INSUFFICIENT_DATA",
                "confidence": 0,
                "signal": "Yetersiz veri"
            }

        # Count buy vs sell trades
        buy_count = sum(1 for t in whale_trades if t['side'] == 'BUY')
        sell_count = len(whale_trades) - buy_count

        total_trades = len(whale_trades)
        buy_ratio = buy_count / total_trades

        if buy_ratio > 0.7:
            pattern = "ACCUMULATION"
            confidence = min(95, 50 + (buy_ratio * 50))
            signal = "🟢 Whale Accumulation - Güçlü alım birikimi tespit edildi"
        elif buy_ratio < 0.3:
            pattern = "DISTRIBUTION"
            confidence = min(95, 50 + ((1 - buy_ratio) * 50))
            signal = "🔴 Whale Distribution - Güçlü satım dağılımı tespit edildi"
        else:
            pattern = "NEUTRAL"
            confidence = 50
            signal = "🟡 Neutral - Net bir yön tespit edilmedi"

        return {
            "pattern": pattern,
            "confidence": round(confidence, 2),
            "signal": signal,
            "buy_count": buy_count,
            "sell_count": sell_count
        }

    def analyze(self, symbol):
        """Complete whale activity analysis"""
        print(f"[Whale] Analyzing {symbol}...")

        # Get data
        current_price = self.get_current_price(symbol)
        trades = self.get_recent_trades(symbol, limit=500)

        if not trades:
            return {
                "symbol": symbol,
                "current_price": current_price,
                "whale_activity": {
                    "detected": False,
                    "message": "Veri alınamadı"
                },
                "timestamp": datetime.now().isoformat()
            }

        # Analyze trades
        analysis = self.analyze_trades(trades, current_price)

        # Calculate pressure
        pressure = self.calculate_pressure(
            analysis['whale_buy_volume'],
            analysis['whale_sell_volume']
        )

        # Detect patterns
        accumulation = self.detect_accumulation(analysis['whale_trades'])

        # Overall whale activity status
        whale_detected = analysis['whale_count'] > 0

        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "whale_activity": {
                "detected": whale_detected,
                "whale_count": analysis['whale_count'],
                "total_volume": analysis['total_whale_volume'],
                "buy_volume": analysis['whale_buy_volume'],
                "sell_volume": analysis['whale_sell_volume'],
                "volume_spike": analysis['volume_spike'],
                "avg_trade_size": analysis['avg_trade_size']
            },
            "pressure": pressure,
            "accumulation": accumulation,
            "recent_whale_trades": analysis['whale_trades'][:5],
            "recommendations": self._generate_recommendations(pressure, accumulation, analysis),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_recommendations(self, pressure, accumulation, analysis):
        """Generate trading recommendations"""
        recommendations = []

        # Accumulation signal
        if accumulation['pattern'] == 'ACCUMULATION' and accumulation['confidence'] > 70:
            recommendations.append({
                "action": "BUY_SIGNAL",
                "reason": "Whale accumulation detected",
                "confidence": accumulation['confidence']
            })
        elif accumulation['pattern'] == 'DISTRIBUTION' and accumulation['confidence'] > 70:
            recommendations.append({
                "action": "SELL_SIGNAL",
                "reason": "Whale distribution detected",
                "confidence": accumulation['confidence']
            })

        # Pressure signal
        if pressure['status'] == 'HEAVY_BUYING':
            recommendations.append({
                "action": "FOLLOW_WHALES",
                "reason": "Heavy whale buying pressure",
                "confidence": pressure['buy_pressure']
            })
        elif pressure['status'] == 'HEAVY_SELLING':
            recommendations.append({
                "action": "CAUTION",
                "reason": "Heavy whale selling pressure",
                "confidence": pressure['sell_pressure']
            })

        # Volume spike warning
        if analysis['volume_spike']:
            recommendations.append({
                "action": "HIGH_VOLATILITY",
                "reason": "Unusual volume spike detected",
                "confidence": 85
            })

        if not recommendations:
            recommendations.append({
                "action": "NEUTRAL",
                "reason": "No strong whale signals",
                "confidence": 50
            })

        return recommendations

tracker = WhaleTracker()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "service": "Whale Activity Tracker",
        "status": "healthy",
        "port": 5015,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze/<symbol>', methods=['GET'])
def analyze_symbol(symbol):
    try:
        result = tracker.analyze(symbol.upper())
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Whale] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        results = {}
        for symbol in symbols[:10]:  # Max 10 symbols
            results[symbol] = tracker.analyze(symbol.upper())
        return jsonify({"success": True, "data": results, "count": len(results)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("🐋 Whale Activity Tracker starting...")
    print("📡 Listening on port 5015")
    app.run(host='0.0.0.0', port=5015, debug=False)
