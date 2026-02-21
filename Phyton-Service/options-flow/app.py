"""
📊 OPTIONS FLOW ANALYZER
Deribit options data analysis and gamma squeeze detection
Port: 5018

Features:
- Put/Call ratio analysis
- Open interest tracking
- Implied volatility analysis
- Gamma squeeze detection
- Max pain calculation
- Options sentiment scoring

WHITE-HAT COMPLIANCE: Educational purpose, transparent analysis
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from datetime import datetime, timedelta
import sys

app = Flask(__name__)
CORS(app)
sys.stdout = sys.stderr

class OptionsAnalyzer:
    def __init__(self):
        self.deribit_api = "https://www.deribit.com/api/v2/public"
        self.binance_api = "https://api.binance.com"

    def get_current_price(self, symbol="BTC"):
        """Get current spot price"""
        try:
            if symbol == "BTC":
                binance_symbol = "BTCUSDT"
            elif symbol == "ETH":
                binance_symbol = "ETHUSDT"
            else:
                return 0

            url = f"{self.binance_api}/api/v3/ticker/price"
            params = {"symbol": binance_symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data.get('price', 0))
        except:
            return 0

    def get_instruments(self, currency="BTC", kind="option"):
        """Get available options instruments from Deribit"""
        try:
            url = f"{self.deribit_api}/get_instruments"
            params = {
                "currency": currency,
                "kind": kind,
                "expired": "false"
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get('result'):
                return data['result']
            return []
        except Exception as e:
            print(f"[Options] Instruments error: {e}")
            return []

    def get_option_data(self, instrument_name):
        """Get option data for specific instrument"""
        try:
            url = f"{self.deribit_api}/ticker"
            params = {"instrument_name": instrument_name}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            if data.get('result'):
                return data['result']
            return None
        except:
            return None

    def analyze_put_call_ratio(self, instruments, current_price):
        """Calculate Put/Call ratio from options data"""
        puts_oi = 0
        calls_oi = 0
        puts_volume = 0
        calls_volume = 0

        for instrument in instruments[:50]:  # Limit to 50 instruments for performance
            instrument_name = instrument.get('instrument_name', '')

            # Check if it's a put or call
            is_put = '-P' in instrument_name
            is_call = '-C' in instrument_name

            # Get option data
            option_data = self.get_option_data(instrument_name)
            if not option_data:
                continue

            oi = float(option_data.get('open_interest', 0))
            volume = float(option_data.get('stats', {}).get('volume', 0))

            if is_put:
                puts_oi += oi
                puts_volume += volume
            elif is_call:
                calls_oi += oi
                calls_volume += volume

        # Calculate ratios
        pc_ratio_oi = (puts_oi / calls_oi) if calls_oi > 0 else 0
        pc_ratio_volume = (puts_volume / calls_volume) if calls_volume > 0 else 0

        # Interpretation
        if pc_ratio_oi > 1.0:
            signal = "BEARISH"
            interpretation = "⚠️ Yüksek Put/Call oranı - Bearish sentiment"
        elif pc_ratio_oi > 0.7:
            signal = "NEUTRAL_BEARISH"
            interpretation = "🟡 Orta Put/Call oranı - Nötr/Hafif Bearish"
        elif pc_ratio_oi > 0.5:
            signal = "NEUTRAL"
            interpretation = "⚖️ Dengeli Put/Call oranı - Nötr"
        else:
            signal = "BULLISH"
            interpretation = "🟢 Düşük Put/Call oranı - Bullish sentiment"

        return {
            "puts_oi": round(puts_oi, 2),
            "calls_oi": round(calls_oi, 2),
            "pc_ratio_oi": round(pc_ratio_oi, 4),
            "puts_volume": round(puts_volume, 2),
            "calls_volume": round(calls_volume, 2),
            "pc_ratio_volume": round(pc_ratio_volume, 4),
            "signal": signal,
            "interpretation": interpretation
        }

    def detect_gamma_squeeze(self, instruments, current_price):
        """Detect potential gamma squeeze conditions"""
        # Look for high OI at strikes near current price
        atm_calls = []

        for instrument in instruments[:100]:
            instrument_name = instrument.get('instrument_name', '')

            # Only check calls
            if '-C' not in instrument_name:
                continue

            # Extract strike price from instrument name
            # Format: BTC-31OCT25-110000-C
            try:
                parts = instrument_name.split('-')
                strike = float(parts[2])
            except:
                continue

            # Check if near the money (within 10%)
            if abs(strike - current_price) / current_price > 0.10:
                continue

            option_data = self.get_option_data(instrument_name)
            if not option_data:
                continue

            oi = float(option_data.get('open_interest', 0))
            volume = float(option_data.get('stats', {}).get('volume', 0))

            if oi > 0:
                atm_calls.append({
                    "strike": strike,
                    "oi": oi,
                    "volume": volume
                })

        if not atm_calls:
            return {
                "risk": "LOW",
                "signal": "No significant gamma risk detected",
                "atm_call_oi": 0
            }

        # Calculate total ATM call OI
        total_atm_oi = sum([c['oi'] for c in atm_calls])

        # Gamma squeeze risk levels
        if total_atm_oi > 10000:
            risk = "HIGH"
            signal = "⚠️ Yüksek ATM call OI - Gamma squeeze riski"
        elif total_atm_oi > 5000:
            risk = "MEDIUM"
            signal = "🟡 Orta ATM call OI - Potansiyel gamma baskısı"
        else:
            risk = "LOW"
            signal = "🟢 Düşük gamma riski"

        return {
            "risk": risk,
            "signal": signal,
            "atm_call_oi": round(total_atm_oi, 2),
            "current_price": round(current_price, 2),
            "atm_strikes": len(atm_calls)
        }

    def calculate_implied_volatility_stats(self, instruments, current_price):
        """Calculate IV statistics"""
        iv_values = []

        for instrument in instruments[:50]:
            option_data = self.get_option_data(instrument.get('instrument_name', ''))
            if not option_data:
                continue

            iv = option_data.get('mark_iv')
            if iv and iv > 0:
                iv_values.append(float(iv))

        if not iv_values:
            return {
                "avg_iv": 0,
                "min_iv": 0,
                "max_iv": 0,
                "signal": "NO_DATA"
            }

        avg_iv = sum(iv_values) / len(iv_values)
        min_iv = min(iv_values)
        max_iv = max(iv_values)

        # IV interpretation (in %)
        if avg_iv > 100:
            signal = "EXTREME_VOLATILITY"
            interpretation = "⚠️ Aşırı yüksek IV - Yüksek belirsizlik"
        elif avg_iv > 70:
            signal = "HIGH_VOLATILITY"
            interpretation = "📈 Yüksek IV - Artmış volatilite beklentisi"
        elif avg_iv > 40:
            signal = "NORMAL"
            interpretation = "⚖️ Normal IV seviyeleri"
        else:
            signal = "LOW_VOLATILITY"
            interpretation = "📉 Düşük IV - Düşük volatilite beklentisi"

        return {
            "avg_iv": round(avg_iv, 2),
            "min_iv": round(min_iv, 2),
            "max_iv": round(max_iv, 2),
            "signal": signal,
            "interpretation": interpretation
        }

    def analyze(self, currency="BTC"):
        """Complete options flow analysis"""
        print(f"[Options] Analyzing options flow for {currency}...")

        current_price = self.get_current_price(currency)

        if current_price == 0:
            return {
                "error": "Spot fiyat alınamadı",
                "timestamp": datetime.now().isoformat()
            }

        instruments = self.get_instruments(currency)

        if not instruments:
            return {
                "error": "Options verisi alınamadı (Deribit API)",
                "currency": currency,
                "current_price": round(current_price, 2),
                "timestamp": datetime.now().isoformat(),
                "note": "Gerçek zamanlı options verisi için Deribit API erişimi gerekli"
            }

        # Analyze Put/Call ratio
        pc_analysis = self.analyze_put_call_ratio(instruments, current_price)

        # Detect gamma squeeze
        gamma_analysis = self.detect_gamma_squeeze(instruments, current_price)

        # Calculate IV stats
        iv_stats = self.calculate_implied_volatility_stats(instruments, current_price)

        # Overall sentiment
        sentiment_score = 50  # Neutral baseline

        if pc_analysis['signal'] == "BULLISH":
            sentiment_score += 20
        elif pc_analysis['signal'] == "BEARISH":
            sentiment_score -= 20

        if gamma_analysis['risk'] == "HIGH":
            sentiment_score += 15

        if iv_stats['signal'] == "HIGH_VOLATILITY":
            sentiment_score -= 10

        sentiment_score = max(0, min(100, sentiment_score))

        if sentiment_score >= 70:
            overall_sentiment = "BULLISH"
        elif sentiment_score >= 45:
            overall_sentiment = "NEUTRAL"
        else:
            overall_sentiment = "BEARISH"

        return {
            "currency": currency,
            "current_price": round(current_price, 2),
            "put_call_analysis": pc_analysis,
            "gamma_squeeze_detection": gamma_analysis,
            "implied_volatility": iv_stats,
            "overall_sentiment": {
                "score": round(sentiment_score, 2),
                "signal": overall_sentiment
            },
            "total_instruments": len(instruments),
            "timestamp": datetime.now().isoformat()
        }

analyzer = OptionsAnalyzer()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "service": "Options Flow Analyzer",
        "status": "healthy",
        "port": 5018,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze/<currency>', methods=['GET'])
def analyze(currency):
    try:
        result = analyzer.analyze(currency.upper())
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Options] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/put-call-ratio/<currency>', methods=['GET'])
def put_call_ratio(currency):
    """Get only Put/Call ratio"""
    try:
        current_price = analyzer.get_current_price(currency.upper())
        instruments = analyzer.get_instruments(currency.upper())

        if not instruments:
            return jsonify({"success": False, "error": "Options data unavailable"}), 400

        pc_analysis = analyzer.analyze_put_call_ratio(instruments, current_price)

        return jsonify({
            "success": True,
            "data": {
                "currency": currency.upper(),
                "current_price": round(current_price, 2),
                "put_call_ratio": pc_analysis,
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        print(f"[Options] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("📊 Options Flow Analyzer starting...")
    print("📡 Listening on port 5018")
    app.run(host='0.0.0.0', port=5018, debug=False)
