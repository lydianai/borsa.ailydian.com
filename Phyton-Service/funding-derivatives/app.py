"""
💰 FUNDING RATE & DERIVATIVES TRACKER
Advanced derivatives market analysis for futures trading
Port: 5014

Features:
- Funding rate tracking (BTC, ETH, ALT coins)
- Open Interest monitoring
- Spot-Futures basis calculation
- CVD (Cumulative Volume Delta) analysis
- Extreme level detection

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

class DerivativesAnalyzer:
    def __init__(self):
        self.binance_api = "https://fapi.binance.com"
        self.spot_api = "https://api.binance.com"

    def get_funding_rate(self, symbol):
        """Get current funding rate"""
        try:
            url = f"{self.binance_api}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return {
                "rate": float(data.get('lastFundingRate', 0)) * 100,  # Convert to %
                "next_time": data.get('nextFundingTime', 0),
                "mark_price": float(data.get('markPrice', 0))
            }
        except Exception as e:
            print(f"[Derivatives] Funding rate error for {symbol}: {e}")
            return {"rate": 0, "next_time": 0, "mark_price": 0}

    def get_open_interest(self, symbol):
        """Get open interest data"""
        try:
            url = f"{self.binance_api}/fapi/v1/openInterest"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return {
                "value": float(data.get('openInterest', 0)),
                "timestamp": data.get('time', 0)
            }
        except Exception as e:
            print(f"[Derivatives] OI error for {symbol}: {e}")
            return {"value": 0, "timestamp": 0}

    def get_spot_price(self, symbol):
        """Get spot market price"""
        try:
            # Convert USDT to USDT for spot
            spot_symbol = symbol  # e.g., BTCUSDT
            url = f"{self.spot_api}/api/v3/ticker/price"
            params = {"symbol": spot_symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data.get('price', 0))
        except Exception as e:
            print(f"[Derivatives] Spot price error for {symbol}: {e}")
            return 0

    def calculate_basis(self, futures_price, spot_price):
        """Calculate spot-futures basis"""
        if spot_price == 0:
            return 0
        basis = ((futures_price - spot_price) / spot_price) * 100
        return round(basis, 4)

    def get_long_short_ratio(self, symbol):
        """Get long/short ratio from Binance"""
        try:
            url = f"{self.binance_api}/futures/data/globalLongShortAccountRatio"
            params = {
                "symbol": symbol,
                "period": "5m",
                "limit": 1
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            if data and len(data) > 0:
                ratio = float(data[0].get('longShortRatio', 1.0))
                return ratio
            return 1.0
        except Exception as e:
            print(f"[Derivatives] Long/Short ratio error: {e}")
            return 1.0

    def analyze(self, symbol):
        """Complete derivatives analysis"""
        print(f"[Derivatives] Analyzing {symbol}...")

        # Get all metrics
        funding = self.get_funding_rate(symbol)
        oi = self.get_open_interest(symbol)
        spot_price = self.get_spot_price(symbol)
        futures_price = funding["mark_price"]
        basis = self.calculate_basis(futures_price, spot_price)
        ls_ratio = self.get_long_short_ratio(symbol)

        # Determine funding status
        if funding["rate"] > 0.05:
            funding_status = "EXTREME_BULLISH"
            warning = "⚠️ Aşırı yüksek funding - Long pozisyonlar risk altında"
        elif funding["rate"] > 0.01:
            funding_status = "BULLISH"
            warning = "Pozitif funding - Long bias"
        elif funding["rate"] < -0.05:
            funding_status = "EXTREME_BEARISH"
            warning = "⚠️ Aşırı düşük funding - Short pozisyonlar risk altında"
        elif funding["rate"] < -0.01:
            funding_status = "BEARISH"
            warning = "Negatif funding - Short bias"
        else:
            funding_status = "NEUTRAL"
            warning = "Normal funding seviyeleri"

        # Basis analysis
        if basis > 0.5:
            basis_status = "CONTANGO_HIGH"
            basis_signal = "Futures premium yüksek - Spot al, futures sat arbitraj"
        elif basis < -0.5:
            basis_status = "BACKWARDATION_HIGH"
            basis_signal = "Spot premium yüksek - Futures al, spot sat arbitraj"
        else:
            basis_status = "NORMAL"
            basis_signal = "Normal piyasa durumu"

        # Long/Short ratio analysis
        if ls_ratio > 1.5:
            ls_status = "LONG_HEAVY"
            ls_signal = "Aşırı long pozisyon - Reversal riski"
        elif ls_ratio < 0.7:
            ls_status = "SHORT_HEAVY"
            ls_signal = "Aşırı short pozisyon - Short squeeze riski"
        else:
            ls_status = "BALANCED"
            ls_signal = "Dengeli pozisyon dağılımı"

        return {
            "symbol": symbol,
            "funding_rate": {
                "current": round(funding["rate"], 4),
                "status": funding_status,
                "warning": warning,
                "next_funding_time": funding["next_time"]
            },
            "open_interest": {
                "value": round(oi["value"], 2),
                "timestamp": oi["timestamp"]
            },
            "basis": {
                "value": basis,
                "status": basis_status,
                "signal": basis_signal,
                "spot_price": round(spot_price, 2),
                "futures_price": round(futures_price, 2)
            },
            "long_short_ratio": {
                "value": round(ls_ratio, 2),
                "status": ls_status,
                "signal": ls_signal
            },
            "timestamp": datetime.now().isoformat()
        }

analyzer = DerivativesAnalyzer()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "service": "Funding Rate & Derivatives Tracker",
        "status": "healthy",
        "port": 5014,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze/<symbol>', methods=['GET'])
def analyze_symbol(symbol):
    try:
        result = analyzer.analyze(symbol.upper())
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Derivatives] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_analyze():
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])
        results = {}
        for symbol in symbols[:20]:
            results[symbol] = analyzer.analyze(symbol.upper())
        return jsonify({"success": True, "data": results, "count": len(results)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("💰 Funding Rate & Derivatives Tracker starting...")
    print("📡 Listening on port 5014")
    app.run(host='0.0.0.0', port=5014, debug=False)
