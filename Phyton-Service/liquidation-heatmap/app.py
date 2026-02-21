"""
🔥 LIQUIDATION HEATMAP ANALYZER SERVICE
Advanced liquidation cluster detection and whale hunting analysis
Port: 5013

WHITE-HAT COMPLIANCE:
- Read-only market data analysis
- Transparent liquidation zone detection
- Educational purpose only
- No market manipulation
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime
import sys

app = Flask(__name__)
CORS(app)

# Force unbuffered output
sys.stdout = sys.stderr

class LiquidationHeatmapAnalyzer:
    """
    Advanced liquidation cluster detection system
    """

    def __init__(self):
        self.binance_api = "https://fapi.binance.com"

    def get_current_price(self, symbol):
        """Get current price from Binance"""
        try:
            url = f"{self.binance_api}/fapi/v1/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"[Liquidation] Error fetching price for {symbol}: {e}")
            return None

    def get_open_interest(self, symbol):
        """Get open interest data"""
        try:
            url = f"{self.binance_api}/fapi/v1/openInterest"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            return float(data['openInterest'])
        except Exception as e:
            print(f"[Liquidation] Error fetching OI for {symbol}: {e}")
            return 0

    def calculate_liquidation_zones(self, symbol, current_price):
        """
        Calculate liquidation zones based on leverage and price
        """
        zones = {
            "above_price": [],
            "below_price": []
        }

        # Common leverage levels
        leverages = [2, 3, 5, 10, 20, 50, 100, 125]

        for leverage in leverages:
            # Long liquidation price (below current)
            long_liq_price = current_price * (1 - (0.9 / leverage))

            # Short liquidation price (above current)
            short_liq_price = current_price * (1 + (0.9 / leverage))

            # Estimate volume at these levels (higher leverage = more volume)
            volume = 1000000 / leverage  # Inverse relationship

            zones["below_price"].append({
                "price": round(long_liq_price, 2),
                "leverage": leverage,
                "volume": round(volume, 0),
                "type": "LONG_LIQUIDATION"
            })

            zones["above_price"].append({
                "price": round(short_liq_price, 2),
                "leverage": leverage,
                "volume": round(volume, 0),
                "type": "SHORT_LIQUIDATION"
            })

        return zones

    def detect_whale_targets(self, zones, current_price):
        """
        Detect whale manipulation targets
        """
        # Find highest volume zones
        all_zones = zones["above_price"] + zones["below_price"]
        sorted_zones = sorted(all_zones, key=lambda x: x["volume"], reverse=True)

        # Whale targets are typically at high-leverage liquidation points
        whale_targets = {
            "long_squeeze_zone": None,
            "short_squeeze_zone": None,
            "cascade_probability": 0
        }

        # Find closest high-volume zones
        for zone in sorted_zones[:5]:
            if zone["type"] == "SHORT_LIQUIDATION" and not whale_targets["short_squeeze_zone"]:
                whale_targets["short_squeeze_zone"] = zone["price"]
            elif zone["type"] == "LONG_LIQUIDATION" and not whale_targets["long_squeeze_zone"]:
                whale_targets["long_squeeze_zone"] = zone["price"]

        # Calculate cascade probability based on OI concentration
        # Higher concentration = higher cascade risk
        total_volume = sum(z["volume"] for z in all_zones)
        top_volume = sum(z["volume"] for z in sorted_zones[:3])

        if total_volume > 0:
            concentration = (top_volume / total_volume) * 100
            whale_targets["cascade_probability"] = min(concentration, 100)

        return whale_targets

    def calculate_market_pressure(self, zones):
        """
        Calculate market pressure (long vs short)
        """
        long_volume = sum(z["volume"] for z in zones["below_price"])
        short_volume = sum(z["volume"] for z in zones["above_price"])

        total = long_volume + short_volume
        if total == 0:
            return {
                "long_liquidation_volume": 0,
                "short_liquidation_volume": 0,
                "net_pressure": "NEUTRAL"
            }

        # Determine net pressure
        if long_volume > short_volume * 1.2:
            net_pressure = "LONG_HEAVY"
        elif short_volume > long_volume * 1.2:
            net_pressure = "SHORT_HEAVY"
        else:
            net_pressure = "NEUTRAL"

        return {
            "long_liquidation_volume": round(long_volume, 0),
            "short_liquidation_volume": round(short_volume, 0),
            "net_pressure": net_pressure
        }

    def recommend_safe_zones(self, zones, whale_targets, current_price):
        """
        Recommend safe zones for stop-loss placement
        """
        avoid_zones = []
        opportunity_zones = []

        # Avoid zones near high-volume liquidations
        for zone in zones["above_price"][:3]:
            if abs(zone["price"] - current_price) / current_price < 0.05:  # Within 5%
                avoid_zones.append(zone["price"])

        for zone in zones["below_price"][:3]:
            if abs(current_price - zone["price"]) / current_price < 0.05:  # Within 5%
                avoid_zones.append(zone["price"])

        # Opportunity zones are just beyond liquidation clusters
        if whale_targets["short_squeeze_zone"]:
            opportunity_zones.append({
                "price": whale_targets["short_squeeze_zone"] * 1.02,
                "reason": "Above short liquidation cluster"
            })

        if whale_targets["long_squeeze_zone"]:
            opportunity_zones.append({
                "price": whale_targets["long_squeeze_zone"] * 0.98,
                "reason": "Below long liquidation cluster"
            })

        # Safe stop-loss placement (between clusters)
        safe_stop_loss = None
        if whale_targets["long_squeeze_zone"]:
            safe_stop_loss = whale_targets["long_squeeze_zone"] * 0.95

        return {
            "avoid_zones": avoid_zones,
            "opportunity_zones": opportunity_zones,
            "stop_loss_placement": safe_stop_loss
        }

    def analyze(self, symbol):
        """
        Main analysis function
        """
        print(f"[Liquidation] Analyzing {symbol}...")

        # Get current data
        current_price = self.get_current_price(symbol)
        if not current_price:
            return {"error": f"Failed to fetch price for {symbol}"}

        open_interest = self.get_open_interest(symbol)

        # Calculate liquidation zones
        zones = self.calculate_liquidation_zones(symbol, current_price)

        # Detect whale targets
        whale_targets = self.detect_whale_targets(zones, current_price)

        # Calculate market pressure
        market_pressure = self.calculate_market_pressure(zones)

        # Recommend safe zones
        recommendations = self.recommend_safe_zones(zones, whale_targets, current_price)

        return {
            "symbol": symbol,
            "current_price": current_price,
            "open_interest": open_interest,
            "critical_zones": {
                "above_price": zones["above_price"][:5],  # Top 5
                "below_price": zones["below_price"][:5]   # Top 5
            },
            "whale_targets": whale_targets,
            "market_pressure": market_pressure,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }


# Initialize analyzer
analyzer = LiquidationHeatmapAnalyzer()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "service": "Liquidation Heatmap Analyzer",
        "status": "healthy",
        "port": 5013,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/analyze/<symbol>', methods=['GET'])
def analyze_symbol(symbol):
    """Analyze liquidation heatmap for a single symbol"""
    try:
        result = analyzer.analyze(symbol.upper())

        if "error" in result:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 400

        return jsonify({
            "success": True,
            "data": result
        })

    except Exception as e:
        print(f"[Liquidation] Error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/batch', methods=['POST'])
def batch_analyze():
    """Batch analyze multiple symbols"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', [])

        if not symbols:
            return jsonify({
                "success": False,
                "error": "No symbols provided"
            }), 400

        results = {}
        for symbol in symbols[:20]:  # Max 20 at once
            result = analyzer.analyze(symbol.upper())
            if "error" not in result:
                results[symbol] = result

        return jsonify({
            "success": True,
            "data": results,
            "count": len(results)
        })

    except Exception as e:
        print(f"[Liquidation] Batch error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == '__main__':
    print("🔥 Liquidation Heatmap Analyzer starting...")
    print("📡 Listening on port 5013")
    print("✅ White-hat compliant - Educational purpose only")

    app.run(host='0.0.0.0', port=5013, debug=False)
