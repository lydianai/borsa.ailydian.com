"""
FEATURE-ENHANCED PREDICTOR
Handles predictions with 150+ features from Feature Engineering Service
Lightweight implementation - works without PyTorch
"""

import numpy as np
from typing import Dict, List, Any, Optional


class FeatureEnhancedPredictor:
    """
    Lightweight predictor that works with 150+ features
    Uses ensemble voting and statistical analysis
    """

    def __init__(self):
        self.feature_count = 0
        self.prediction_history = []

    def predict_from_features(
        self,
        feature_data: List[Dict[str, float]],
        symbol: str,
        interval: str = '1h'
    ) -> Dict[str, Any]:
        """
        Make prediction using 150+ features

        Args:
            feature_data: List of feature dictionaries from Feature Engineering
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe (e.g., '1h', '4h')

        Returns:
            Dict with prediction, confidence, reasoning
        """

        if not feature_data or len(feature_data) == 0:
            return self._fallback_prediction(symbol, "No feature data provided")

        # Extract features from last candle
        latest_features = feature_data[-1] if isinstance(feature_data, list) else feature_data

        # Count features
        self.feature_count = len(latest_features)

        # Analyze key features
        analysis = self._analyze_features(latest_features)

        # Generate signal
        signal = self._generate_signal(analysis)

        # Calculate confidence
        confidence = self._calculate_confidence(analysis)

        # Generate reasoning
        reasoning = self._generate_reasoning(analysis, signal)

        return {
            'symbol': symbol,
            'prediction': {
                'signal': signal,
                'confidence': confidence,
                'strength': min(10, max(1, int(confidence * 10)))
            },
            'current_price': latest_features.get('close', 0),
            'reasoning': reasoning,
            'models_used': ['feature_enhanced_ensemble'],
            'score': confidence,
            'feature_count': self.feature_count,
            'analysis': {
                'bullish_signals': analysis['bullish_count'],
                'bearish_signals': analysis['bearish_count'],
                'neutral_signals': analysis['neutral_count'],
                'total_signals': analysis['total_signals']
            }
        }

    def _analyze_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze 150+ features to generate trading signals
        """
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        # RSI Analysis (multiple timeframes)
        for key in features.keys():
            if 'rsi' in key.lower():
                rsi_value = features[key]
                if rsi_value < 30:
                    bullish_count += 2  # Oversold = bullish
                elif rsi_value > 70:
                    bearish_count += 2  # Overbought = bearish
                else:
                    neutral_count += 1

        # MACD Analysis
        for key in features.keys():
            if 'macd_hist' in key.lower():
                macd_hist = features[key]
                if macd_hist > 0:
                    bullish_count += 1
                elif macd_hist < 0:
                    bearish_count += 1

        # Bollinger Bands Analysis
        for key in features.keys():
            if 'bb_percent' in key.lower():
                bb_percent = features[key]
                if bb_percent < 0.2:
                    bullish_count += 1  # Near lower band
                elif bb_percent > 0.8:
                    bearish_count += 1  # Near upper band

        # Moving Average Analysis
        price = features.get('close', 0)
        for key in features.keys():
            if 'price_to_sma' in key.lower() or 'price_to_ema' in key.lower():
                ma_ratio = features[key]
                if ma_ratio > 1.02:  # Price 2% above MA
                    bullish_count += 1
                elif ma_ratio < 0.98:  # Price 2% below MA
                    bearish_count += 1

        # Volume Analysis
        for key in features.keys():
            if 'volume_ratio' in key.lower():
                vol_ratio = features[key]
                if vol_ratio > 1.5:  # High volume
                    # Volume confirms trend
                    if bullish_count > bearish_count:
                        bullish_count += 2
                    elif bearish_count > bullish_count:
                        bearish_count += 2

        # ADX (Trend Strength)
        for key in features.keys():
            if 'adx' in key.lower() and 'plus_di' not in key.lower() and 'minus_di' not in key.lower():
                adx_value = features[key]
                if adx_value > 25:  # Strong trend
                    # Check DI difference
                    for di_key in features.keys():
                        if 'di_diff' in di_key.lower():
                            di_diff = features[di_key]
                            if di_diff > 5:
                                bullish_count += 2
                            elif di_diff < -5:
                                bearish_count += 2

        # Pattern Recognition
        for key in features.keys():
            if any(pattern in key.lower() for pattern in ['hammer', 'morning_star', 'three_white_soldiers']):
                if features[key] != 0:  # Pattern detected
                    bullish_count += 3
            elif any(pattern in key.lower() for pattern in ['shooting_star', 'evening_star', 'three_black_crows']):
                if features[key] != 0:  # Pattern detected
                    bearish_count += 3
            elif any(pattern in key.lower() for pattern in ['doji', 'engulfing']):
                if features[key] != 0:
                    neutral_count += 1

        # Momentum Analysis
        for key in features.keys():
            if 'momentum_pct' in key.lower():
                momentum = features[key]
                if momentum > 2:  # Strong positive momentum
                    bullish_count += 1
                elif momentum < -2:  # Strong negative momentum
                    bearish_count += 1

        # Stochastic Oscillator
        for key in features.keys():
            if 'stoch_k' in key.lower():
                stoch_k = features[key]
                if stoch_k < 20:
                    bullish_count += 1  # Oversold
                elif stoch_k > 80:
                    bearish_count += 1  # Overbought

        total_signals = bullish_count + bearish_count + neutral_count

        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_signals': max(1, total_signals)
        }

    def _generate_signal(self, analysis: Dict[str, Any]) -> str:
        """
        Generate BUY/SELL/HOLD signal based on analysis
        """
        bullish = analysis['bullish_count']
        bearish = analysis['bearish_count']
        total = analysis['total_signals']

        # Calculate percentages
        bullish_pct = (bullish / total) * 100 if total > 0 else 0
        bearish_pct = (bearish / total) * 100 if total > 0 else 0

        # Decision thresholds
        if bullish_pct >= 60:
            return 'BUY'
        elif bearish_pct >= 60:
            return 'SELL'
        elif bullish_pct > bearish_pct and bullish_pct >= 40:
            return 'BUY'
        elif bearish_pct > bullish_pct and bearish_pct >= 40:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """
        Calculate prediction confidence (0.0 - 1.0)
        """
        bullish = analysis['bullish_count']
        bearish = analysis['bearish_count']
        total = analysis['total_signals']

        if total == 0:
            return 0.5

        # Confidence is based on signal agreement
        max_signals = max(bullish, bearish)
        confidence = (max_signals / total)

        # Boost confidence if strong agreement
        if confidence > 0.7:
            confidence = min(0.95, confidence * 1.1)

        # Lower confidence if signals are mixed
        if abs(bullish - bearish) < (total * 0.2):
            confidence *= 0.8

        return round(confidence, 2)

    def _generate_reasoning(self, analysis: Dict[str, Any], signal: str) -> str:
        """
        Generate human-readable reasoning for the prediction
        """
        bullish = analysis['bullish_count']
        bearish = analysis['bearish_count']
        neutral = analysis['neutral_count']
        total = analysis['total_signals']

        if signal == 'BUY':
            return f"Strong bullish signals ({bullish}/{total}) from {self.feature_count} features. " \
                   f"Technical indicators show oversold conditions, positive momentum, and bullish patterns."
        elif signal == 'SELL':
            return f"Strong bearish signals ({bearish}/{total}) from {self.feature_count} features. " \
                   f"Technical indicators show overbought conditions, negative momentum, and bearish patterns."
        else:
            return f"Mixed signals ({bullish} bullish, {bearish} bearish, {neutral} neutral) from {self.feature_count} features. " \
                   f"Market consolidating - waiting for clearer direction."

    def _fallback_prediction(self, symbol: str, reason: str) -> Dict[str, Any]:
        """
        Fallback prediction when features are unavailable
        """
        return {
            'symbol': symbol,
            'prediction': {
                'signal': 'HOLD',
                'confidence': 0.5,
                'strength': 5
            },
            'current_price': 0,
            'reasoning': f"Fallback prediction: {reason}",
            'models_used': ['fallback'],
            'score': 0.5,
            'feature_count': 0,
            'analysis': {
                'bullish_signals': 0,
                'bearish_signals': 0,
                'neutral_signals': 1,
                'total_signals': 1
            }
        }
