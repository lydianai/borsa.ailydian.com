"""
📈 MACRO CORRELATION MATRIX
BTC/Altcoin correlation analysis and market regime detection
Port: 5016

Features:
- BTC/Altcoin correlation calculation
- BTC Dominance tracking
- Risk-on / Risk-off detection
- Market divergence analysis
- Correlation heatmap data

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

class CorrelationAnalyzer:
    def __init__(self):
        self.binance_api = "https://api.binance.com"
        # Major altcoins for correlation analysis
        self.altcoins = [
            "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
            "XRPUSDT", "DOTUSDT", "MATICUSDT", "AVAXUSDT"
        ]

    def get_klines(self, symbol, interval='1d', limit=30):
        """Get historical kline data"""
        try:
            url = f"{self.binance_api}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            # Extract close prices
            closes = [float(candle[4]) for candle in data]
            return closes
        except Exception as e:
            print(f"[Correlation] Klines error for {symbol}: {e}")
            return []

    def get_btc_dominance(self):
        """Calculate BTC dominance approximation"""
        try:
            # Get BTC market cap data from ticker
            btc_volume = self.get_24h_volume("BTCUSDT")
            total_volume = btc_volume

            # Add major altcoins
            for alt in self.altcoins:
                alt_volume = self.get_24h_volume(alt)
                total_volume += alt_volume

            if total_volume == 0:
                return 0

            dominance = (btc_volume / total_volume) * 100
            return round(dominance, 2)
        except Exception as e:
            print(f"[Correlation] BTC dominance error: {e}")
            return 0

    def get_24h_volume(self, symbol):
        """Get 24h quote volume"""
        try:
            url = f"{self.binance_api}/api/v3/ticker/24hr"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data.get('quoteVolume', 0))
        except:
            return 0

    def calculate_correlation(self, btc_prices, alt_prices):
        """Calculate Pearson correlation coefficient"""
        if len(btc_prices) < 10 or len(alt_prices) < 10:
            return 0

        if len(btc_prices) != len(alt_prices):
            min_len = min(len(btc_prices), len(alt_prices))
            btc_prices = btc_prices[-min_len:]
            alt_prices = alt_prices[-min_len:]

        # Calculate returns
        btc_returns = [(btc_prices[i] - btc_prices[i-1]) / btc_prices[i-1]
                       for i in range(1, len(btc_prices))]
        alt_returns = [(alt_prices[i] - alt_prices[i-1]) / alt_prices[i-1]
                       for i in range(1, len(alt_prices))]

        if not btc_returns or not alt_returns:
            return 0

        # Pearson correlation
        n = len(btc_returns)
        sum_btc = sum(btc_returns)
        sum_alt = sum(alt_returns)
        sum_btc_sq = sum([r**2 for r in btc_returns])
        sum_alt_sq = sum([r**2 for r in alt_returns])
        sum_product = sum([btc_returns[i] * alt_returns[i] for i in range(n)])

        numerator = n * sum_product - sum_btc * sum_alt
        denominator = ((n * sum_btc_sq - sum_btc**2) * (n * sum_alt_sq - sum_alt**2)) ** 0.5

        if denominator == 0:
            return 0

        correlation = numerator / denominator
        return round(correlation, 4)

    def analyze_market_regime(self, correlations, btc_dominance):
        """Determine market regime (Risk-on / Risk-off)"""

        # Calculate average correlation
        valid_corrs = [c for c in correlations.values() if c != 0]
        if not valid_corrs:
            avg_correlation = 0
        else:
            avg_correlation = sum(valid_corrs) / len(valid_corrs)

        # High correlation + high dominance = Risk-off (fear)
        # High correlation + low dominance = Risk-on (greed)
        # Low correlation = Market confusion

        if avg_correlation > 0.7:
            if btc_dominance > 50:
                regime = "RISK_OFF"
                signal = "⚠️ Risk-Off Modu - BTC'ye kaçış, altcoin'ler düşüş riski"
                recommendation = "BTC tercih et, altcoin pozisyonlarını azalt"
            else:
                regime = "RISK_ON"
                signal = "🟢 Risk-On Modu - Yüksek korelasyon, altseason potansiyeli"
                recommendation = "Altcoin pozisyonları için uygun ortam"
        elif avg_correlation > 0.4:
            regime = "NEUTRAL"
            signal = "🟡 Nötr Mod - Dengeli piyasa"
            recommendation = "Seçici ol, güçlü projelere odaklan"
        else:
            regime = "DECOUPLING"
            signal = "🔀 Decoupling - Düşük korelasyon, bağımsız hareketler"
            recommendation = "Altcoin'ler BTC'den bağımsız hareket edebilir"

        return {
            "regime": regime,
            "signal": signal,
            "recommendation": recommendation,
            "avg_correlation": round(avg_correlation, 4),
            "btc_dominance": btc_dominance
        }

    def detect_divergences(self, btc_prices, correlations):
        """Detect price divergences"""
        divergences = []

        # BTC trend
        btc_trend = "UP" if btc_prices[-1] > btc_prices[0] else "DOWN"

        for symbol, corr in correlations.items():
            if abs(corr) < 0.3:  # Low correlation = potential divergence
                divergences.append({
                    "symbol": symbol,
                    "correlation": corr,
                    "type": "WEAK_CORRELATION",
                    "signal": f"{symbol} BTC'den bağımsız hareket ediyor"
                })

        return divergences

    def analyze(self, base_symbol="BTCUSDT"):
        """Complete correlation analysis"""
        print(f"[Correlation] Analyzing market correlations...")

        # Get BTC prices
        btc_prices = self.get_klines(base_symbol, interval='1d', limit=30)

        if not btc_prices:
            return {
                "error": "BTC fiyat verileri alınamadı",
                "timestamp": datetime.now().isoformat()
            }

        # Calculate correlations with altcoins
        correlations = {}
        for altcoin in self.altcoins:
            alt_prices = self.get_klines(altcoin, interval='1d', limit=30)
            if alt_prices:
                corr = self.calculate_correlation(btc_prices, alt_prices)
                correlations[altcoin] = corr

        # Get BTC dominance
        btc_dominance = self.get_btc_dominance()

        # Analyze market regime
        market_regime = self.analyze_market_regime(correlations, btc_dominance)

        # Detect divergences
        divergences = self.detect_divergences(btc_prices, correlations)

        # Sort correlations by value
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return {
            "base_symbol": base_symbol,
            "base_price": round(btc_prices[-1], 2),
            "correlations": dict(sorted_correlations),
            "market_regime": market_regime,
            "divergences": divergences,
            "strongest_correlation": {
                "symbol": sorted_correlations[0][0] if sorted_correlations else None,
                "value": sorted_correlations[0][1] if sorted_correlations else 0
            },
            "weakest_correlation": {
                "symbol": sorted_correlations[-1][0] if sorted_correlations else None,
                "value": sorted_correlations[-1][1] if sorted_correlations else 0
            },
            "timestamp": datetime.now().isoformat()
        }

analyzer = CorrelationAnalyzer()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "service": "Macro Correlation Matrix",
        "status": "healthy",
        "port": 5016,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        base_symbol = request.args.get('base', 'BTCUSDT')
        result = analyzer.analyze(base_symbol.upper())
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Correlation] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/correlations/<symbol>', methods=['GET'])
def get_correlation(symbol):
    """Get correlation for a specific symbol pair"""
    try:
        base_symbol = request.args.get('base', 'BTCUSDT')

        btc_prices = analyzer.get_klines(base_symbol, interval='1d', limit=30)
        alt_prices = analyzer.get_klines(symbol.upper(), interval='1d', limit=30)

        if not btc_prices or not alt_prices:
            return jsonify({"success": False, "error": "Price data unavailable"}), 400

        correlation = analyzer.calculate_correlation(btc_prices, alt_prices)

        return jsonify({
            "success": True,
            "data": {
                "base": base_symbol,
                "target": symbol.upper(),
                "correlation": correlation,
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        print(f"[Correlation] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("📈 Macro Correlation Matrix starting...")
    print("📡 Listening on port 5016")
    app.run(host='0.0.0.0', port=5016, debug=False)
