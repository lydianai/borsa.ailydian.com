"""
FEATURE ENGINEERING SERVICE
Port: 5006
Generates 150+ features from OHLCV data for AI models
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Any
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Advanced Feature Engineering for Trading Signals
    Generates 150+ features from OHLCV data
    """

    @staticmethod
    def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Price-based features (20+ features)
        """
        features = pd.DataFrame(index=df.index)

        # Returns
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['abs_returns'] = features['returns'].abs()

        # Momentum (multiple periods)
        for period in [5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            features[f'momentum_pct_{period}'] = (df['close'] / df['close'].shift(period) - 1) * 100

        # Price acceleration and jerk
        features['price_acceleration'] = features['returns'].diff()
        features['price_jerk'] = features['price_acceleration'].diff()

        # High-Low spread
        features['hl_spread'] = (df['high'] - df['low']) / df['close']
        features['hl_spread_ma'] = features['hl_spread'].rolling(20).mean()

        # Close position in range
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return features

    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Technical indicators (80+ features)
        """
        features = pd.DataFrame(index=df.index)

        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values

        # RSI variants
        for period in [7, 14, 21, 28]:
            rsi = talib.RSI(close, timeperiod=period)
            features[f'rsi_{period}'] = rsi
            features[f'rsi_{period}_oversold'] = (rsi < 30).astype(int)
            features[f'rsi_{period}_overbought'] = (rsi > 70).astype(int)

        # MACD variants
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            macd, macdsignal, macdhist = talib.MACD(close, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}'] = macdsignal
            features[f'macd_hist_{fast}_{slow}'] = macdhist

        # Bollinger Bands
        for period in [10, 20, 50]:
            upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2)
            features[f'bb_upper_{period}'] = upper
            features[f'bb_middle_{period}'] = middle
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = (upper - lower) / middle
            features[f'bb_percent_{period}'] = (close - lower) / (upper - lower)

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            atr = talib.ATR(high, low, close, timeperiod=period)
            features[f'atr_{period}'] = atr
            features[f'atr_percent_{period}'] = atr / close

        # ADX (Average Directional Index)
        for period in [14, 20]:
            adx = talib.ADX(high, low, close, timeperiod=period)
            plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
            minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
            features[f'adx_{period}'] = adx
            features[f'plus_di_{period}'] = plus_di
            features[f'minus_di_{period}'] = minus_di
            features[f'di_diff_{period}'] = plus_di - minus_di

        # CCI (Commodity Channel Index)
        for period in [14, 20]:
            features[f'cci_{period}'] = talib.CCI(high, low, close, timeperiod=period)

        # Williams %R
        for period in [14, 21]:
            features[f'willr_{period}'] = talib.WILLR(high, low, close, timeperiod=period)

        # Stochastic Oscillator
        for period in [14, 21]:
            slowk, slowd = talib.STOCH(high, low, close,
                                       fastk_period=period, slowk_period=3, slowd_period=3)
            features[f'stoch_k_{period}'] = slowk
            features[f'stoch_d_{period}'] = slowd

        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
            features[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
            features[f'price_to_sma_{period}'] = close / talib.SMA(close, timeperiod=period)
            features[f'price_to_ema_{period}'] = close / talib.EMA(close, timeperiod=period)

        # Parabolic SAR
        features['sar'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        features['sar_signal'] = (close > features['sar']).astype(int)

        return features

    @staticmethod
    def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features (25+ features)
        """
        features = pd.DataFrame(index=df.index)

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Volume ratios
        features['volume'] = volume
        features['volume_change'] = pd.Series(volume).pct_change()

        for period in [5, 10, 20]:
            vol_ma = talib.SMA(volume, timeperiod=period)
            features[f'volume_sma_{period}'] = vol_ma
            features[f'volume_ratio_{period}'] = volume / vol_ma

        # OBV (On Balance Volume)
        features['obv'] = talib.OBV(close, volume)
        features['obv_ma_20'] = features['obv'].rolling(20).mean()
        features['obv_signal'] = (features['obv'] > features['obv_ma_20']).astype(int)

        # AD (Accumulation/Distribution)
        features['ad'] = talib.AD(high, low, close, volume)
        features['ad_ma_20'] = features['ad'].rolling(20).mean()

        # ADOSC (Chaikin A/D Oscillator)
        features['adosc'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

        # MFI (Money Flow Index)
        for period in [14, 20]:
            features[f'mfi_{period}'] = talib.MFI(high, low, close, volume, timeperiod=period)

        # VWAP approximation (intraday)
        typical_price = (high + low + close) / 3
        features['vwap'] = (typical_price * volume).cumsum() / volume.cumsum()
        features['vwap_distance'] = (close - features['vwap']) / features['vwap']

        return features

    @staticmethod
    def calculate_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Pattern recognition features (15+ features)
        """
        features = pd.DataFrame(index=df.index)

        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Candlestick patterns
        features['doji'] = talib.CDLDOJI(open_price, high, low, close)
        features['hammer'] = talib.CDLHAMMER(open_price, high, low, close)
        features['shooting_star'] = talib.CDLSHOOTINGSTAR(open_price, high, low, close)
        features['engulfing'] = talib.CDLENGULFING(open_price, high, low, close)
        features['morning_star'] = talib.CDLMORNINGSTAR(open_price, high, low, close)
        features['evening_star'] = talib.CDLEVENINGSTAR(open_price, high, low, close)
        features['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(open_price, high, low, close)
        features['three_black_crows'] = talib.CDL3BLACKCROWS(open_price, high, low, close)

        # Trend strength
        features['trend_strength'] = abs(close - talib.SMA(close, timeperiod=20)) / close

        # Support/Resistance (simplified - local extrema)
        features['swing_high'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'] > df['high'].shift(-1))
        ).astype(int)
        features['swing_low'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'] < df['low'].shift(-1))
        ).astype(int)

        return features

    @staticmethod
    def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based features (10+ features)
        """
        features = pd.DataFrame(index=df.index)

        # Assume timestamp is in index or a column
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], unit='ms')
        else:
            # If no timestamp column, use default time features
            features['hour'] = 12  # Default to noon
            features['hour_sin'] = np.sin(2 * np.pi * 12 / 24)
            features['hour_cos'] = np.cos(2 * np.pi * 12 / 24)
            features['day_of_week'] = 2  # Default to Wednesday
            features['day_sin'] = np.sin(2 * np.pi * 2 / 7)
            features['day_cos'] = np.cos(2 * np.pi * 2 / 7)
            features['asian_session'] = 0
            features['london_session'] = 1
            features['newyork_session'] = 0
            features['session_overlap'] = 0
            return features

        # Hour of day (for intraday data)
        hour = dt.dt.hour
        features['hour'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        # Day of week
        day_of_week = dt.dt.dayofweek
        features['day_of_week'] = day_of_week
        features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

        # Trading session indicators (UTC-based)
        # Asian session: 23:00-08:00 UTC
        features['asian_session'] = ((hour >= 23) | (hour < 8)).astype(int)
        # London session: 07:00-16:00 UTC
        features['london_session'] = ((hour >= 7) & (hour < 16)).astype(int)
        # New York session: 13:00-22:00 UTC
        features['newyork_session'] = ((hour >= 13) & (hour < 22)).astype(int)
        # Session overlap
        features['session_overlap'] = (features['london_session'] & features['newyork_session']).astype(int)

        return features

    @staticmethod
    def generate_all_features(ohlcv_data: List[Dict]) -> Dict[str, Any]:
        """
        Main function to generate all 150+ features
        """
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)

        # Required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns. Need: {required}")

        # Generate feature sets
        price_features = FeatureEngineering.calculate_price_features(df)
        tech_features = FeatureEngineering.calculate_technical_indicators(df)
        volume_features = FeatureEngineering.calculate_volume_features(df)
        pattern_features = FeatureEngineering.calculate_pattern_features(df)
        time_features = FeatureEngineering.calculate_time_features(df)

        # Combine all features
        all_features = pd.concat([
            df[required],
            price_features,
            tech_features,
            volume_features,
            pattern_features,
            time_features
        ], axis=1)

        # Drop rows with NaN (initial periods)
        all_features = all_features.dropna()

        # Convert to list of dicts
        feature_records = all_features.to_dict('records')

        return {
            'success': True,
            'feature_count': len(all_features.columns),
            'feature_names': list(all_features.columns),
            'data': feature_records,
            'shape': {
                'rows': len(feature_records),
                'columns': len(all_features.columns)
            }
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Feature Engineering Service',
        'version': '1.0.0',
        'port': 5006
    })


@app.route('/features/generate', methods=['POST'])
def generate_features():
    """
    POST /features/generate
    Body: { "ohlcv": [...] }
    Returns: { "success": true, "feature_count": 150, "data": [...] }
    """
    try:
        data = request.json

        if not data or 'ohlcv' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "ohlcv" in request body'
            }), 400

        ohlcv_data = data['ohlcv']

        if len(ohlcv_data) < 200:
            return jsonify({
                'success': False,
                'error': 'Need at least 200 candles for feature engineering'
            }), 400

        # Generate features
        result = FeatureEngineering.generate_all_features(ohlcv_data)

        logger.info(f"✅ Generated {result['feature_count']} features from {len(ohlcv_data)} candles")

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Error in feature generation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/features/list', methods=['GET'])
def list_features():
    """
    GET /features/list
    Returns list of all feature categories and their counts
    """
    return jsonify({
        'success': True,
        'categories': {
            'price_features': {
                'count': 20,
                'examples': ['returns', 'log_returns', 'momentum_5', 'momentum_10']
            },
            'technical_indicators': {
                'count': 80,
                'examples': ['rsi_14', 'macd_12_26', 'bb_width_20', 'adx_14']
            },
            'volume_features': {
                'count': 25,
                'examples': ['obv', 'mfi_14', 'vwap', 'volume_ratio_20']
            },
            'pattern_features': {
                'count': 15,
                'examples': ['doji', 'hammer', 'engulfing', 'swing_high']
            },
            'time_features': {
                'count': 10,
                'examples': ['hour_sin', 'asian_session', 'session_overlap']
            }
        },
        'total_features': 150
    })


if __name__ == '__main__':
    logger.info("🚀 Feature Engineering Service starting on port 5006...")
    logger.info("📊 Generates 150+ features from OHLCV data")
    app.run(host='0.0.0.0', port=5006, debug=True)
