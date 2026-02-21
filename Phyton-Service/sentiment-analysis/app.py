"""
📊 SENTIMENT ANALYSIS ENGINE
Multi-source market sentiment aggregation and analysis
Port: 5017

Features:
- Fear & Greed Index integration
- Social media sentiment scoring
- News sentiment analysis
- Composite sentiment calculation
- Sentiment trend detection

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

class SentimentAnalyzer:
    def __init__(self):
        self.binance_api = "https://api.binance.com"
        self.fear_greed_api = "https://api.alternative.me/fng/"

    def get_fear_greed_index(self):
        """Get Fear & Greed Index from Alternative.me"""
        try:
            response = requests.get(f"{self.fear_greed_api}?limit=7", timeout=10)
            data = response.json()

            if data.get('data'):
                current = data['data'][0]
                history = data['data'][:7]

                # Calculate trend
                values = [int(h['value']) for h in history]
                trend = "INCREASING" if values[0] > values[-1] else "DECREASING"

                return {
                    "value": int(current['value']),
                    "classification": current['value_classification'],
                    "trend": trend,
                    "week_avg": round(sum(values) / len(values), 2),
                    "timestamp": current['timestamp']
                }
            else:
                return self.get_default_fear_greed()

        except Exception as e:
            print(f"[Sentiment] Fear & Greed API error: {e}")
            return self.get_default_fear_greed()

    def get_default_fear_greed(self):
        """Default Fear & Greed when API unavailable"""
        return {
            "value": 50,
            "classification": "Neutral",
            "trend": "STABLE",
            "week_avg": 50,
            "timestamp": datetime.now().timestamp()
        }

    def analyze_price_sentiment(self, symbol="BTCUSDT"):
        """Analyze sentiment from price action"""
        try:
            # Get recent klines
            url = f"{self.binance_api}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": "1h",
                "limit": 24
            }
            response = requests.get(url, params=params, timeout=10)
            klines = response.json()

            if not klines:
                return {"score": 50, "signal": "NEUTRAL"}

            # Calculate price momentum
            opens = [float(k[1]) for k in klines]
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            # Price change
            price_change = ((closes[-1] - opens[0]) / opens[0]) * 100

            # Volume trend
            recent_vol = sum(volumes[-6:]) / 6
            prev_vol = sum(volumes[-12:-6]) / 6
            vol_change = ((recent_vol - prev_vol) / prev_vol) * 100 if prev_vol > 0 else 0

            # Sentiment score (0-100)
            sentiment_score = 50

            # Price influence
            if price_change > 2:
                sentiment_score += 20
            elif price_change > 0:
                sentiment_score += 10
            elif price_change < -2:
                sentiment_score -= 20
            elif price_change < 0:
                sentiment_score -= 10

            # Volume influence
            if vol_change > 20:
                sentiment_score += 10
            elif vol_change < -20:
                sentiment_score -= 10

            # Clamp to 0-100
            sentiment_score = max(0, min(100, sentiment_score))

            # Signal
            if sentiment_score >= 70:
                signal = "BULLISH"
            elif sentiment_score >= 55:
                signal = "MODERATELY_BULLISH"
            elif sentiment_score >= 45:
                signal = "NEUTRAL"
            elif sentiment_score >= 30:
                signal = "MODERATELY_BEARISH"
            else:
                signal = "BEARISH"

            return {
                "score": round(sentiment_score, 2),
                "signal": signal,
                "price_change_24h": round(price_change, 2),
                "volume_change": round(vol_change, 2)
            }

        except Exception as e:
            print(f"[Sentiment] Price analysis error: {e}")
            return {"score": 50, "signal": "NEUTRAL"}

    def analyze_social_sentiment(self):
        """Simulated social media sentiment (placeholder for future API integration)"""
        # In production, this would integrate with Twitter API, Reddit API, etc.
        # For now, we provide a placeholder that returns neutral

        return {
            "twitter_score": 50,
            "reddit_score": 50,
            "telegram_score": 50,
            "overall_score": 50,
            "signal": "NEUTRAL",
            "note": "Social APIs require authentication keys"
        }

    def analyze_news_sentiment(self):
        """Simulated news sentiment (placeholder for future integration)"""
        # In production, this would use NewsAPI, CryptoCompare, or similar

        return {
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "overall_score": 50,
            "signal": "NEUTRAL",
            "note": "News API requires authentication"
        }

    def calculate_composite_sentiment(self, fear_greed, price_sentiment, social, news):
        """Calculate weighted composite sentiment"""

        # Weights for each component
        weights = {
            "fear_greed": 0.35,
            "price": 0.40,
            "social": 0.15,
            "news": 0.10
        }

        # Calculate weighted average
        composite = (
            fear_greed['value'] * weights['fear_greed'] +
            price_sentiment['score'] * weights['price'] +
            social['overall_score'] * weights['social'] +
            news['overall_score'] * weights['news']
        )

        # Determine signal
        if composite >= 75:
            signal = "EXTREME_GREED"
            recommendation = "⚠️ Aşırı açgözlülük - Kâr realizasyonu düşünün"
            risk_level = "HIGH"
        elif composite >= 60:
            signal = "GREED"
            recommendation = "📈 Açgözlülük - Dikkatli ol, pozisyon boyutunu kontrol et"
            risk_level = "MEDIUM_HIGH"
        elif composite >= 45:
            signal = "NEUTRAL"
            recommendation = "⚖️ Nötr piyasa - Seçici olun"
            risk_level = "MEDIUM"
        elif composite >= 30:
            signal = "FEAR"
            recommendation = "📉 Korku - Potansiyel alım fırsatları"
            risk_level = "MEDIUM_LOW"
        else:
            signal = "EXTREME_FEAR"
            recommendation = "💎 Aşırı korku - Güçlü alım fırsatları (dikkatli)"
            risk_level = "LOW"

        return {
            "composite_score": round(composite, 2),
            "signal": signal,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "components": {
                "fear_greed_weight": round(fear_greed['value'] * weights['fear_greed'], 2),
                "price_weight": round(price_sentiment['score'] * weights['price'], 2),
                "social_weight": round(social['overall_score'] * weights['social'], 2),
                "news_weight": round(news['overall_score'] * weights['news'], 2)
            }
        }

    def analyze(self, symbol="BTCUSDT"):
        """Complete sentiment analysis"""
        print(f"[Sentiment] Analyzing market sentiment for {symbol}...")

        # Get all sentiment components
        fear_greed = self.get_fear_greed_index()
        price_sentiment = self.analyze_price_sentiment(symbol)
        social_sentiment = self.analyze_social_sentiment()
        news_sentiment = self.analyze_news_sentiment()

        # Calculate composite
        composite = self.calculate_composite_sentiment(
            fear_greed,
            price_sentiment,
            social_sentiment,
            news_sentiment
        )

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "composite": composite,
            "fear_greed_index": fear_greed,
            "price_sentiment": price_sentiment,
            "social_sentiment": social_sentiment,
            "news_sentiment": news_sentiment,
            "analysis_quality": {
                "fear_greed": "LIVE",
                "price": "LIVE",
                "social": "SIMULATED",
                "news": "SIMULATED"
            }
        }

analyzer = SentimentAnalyzer()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "service": "Sentiment Analysis Engine",
        "status": "healthy",
        "port": 5017,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze', methods=['GET'])
def analyze():
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        result = analyzer.analyze(symbol.upper())
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Sentiment] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/fear-greed', methods=['GET'])
def fear_greed():
    """Get only Fear & Greed Index"""
    try:
        result = analyzer.get_fear_greed_index()
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[Sentiment] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/composite', methods=['GET'])
def composite():
    """Get only composite sentiment score"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        analysis = analyzer.analyze(symbol.upper())
        return jsonify({
            "success": True,
            "data": {
                "symbol": symbol,
                "composite_score": analysis['composite']['composite_score'],
                "signal": analysis['composite']['signal'],
                "recommendation": analysis['composite']['recommendation'],
                "risk_level": analysis['composite']['risk_level'],
                "timestamp": analysis['timestamp']
            }
        })
    except Exception as e:
        print(f"[Sentiment] Error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("📊 Sentiment Analysis Engine starting...")
    print("📡 Listening on port 5017")
    app.run(host='0.0.0.0', port=5017, debug=False)
