"""
QUANTUM LADDER STRATEGY SERVICE
================================
Merdiven misali grafik analizi: ZigZag + Auto Fibonacci + MA 7-25-99
Multi-timeframe confluence detection ile profesyonel destek/direnç tespiti

Author: White-Hat Trading Systems
Date: 2025-11-02
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional
import traceback

app = Flask(__name__)
CORS(app)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Quantum Ladder] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/quantum-ladder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BINANCE_API = 'https://fapi.binance.com/fapi/v1'
FIBONACCI_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
CONFLUENCE_TOLERANCE = 0.002  # 0.2% tolerance for level matching


class ZigZagCalculator:
    """
    ZigZag algoritması - Swing high/low tespiti için
    """

    @staticmethod
    def calculate(prices: np.ndarray, depth: int = 12, deviation: float = 5.0) -> List[Dict]:
        """
        ZigZag noktalarını hesapla

        Args:
            prices: Fiyat dizisi (OHLC)
            depth: Minimum swing mesafesi
            deviation: Minimum % değişim

        Returns:
            List of swing points [{index, price, type: 'high'/'low'}]
        """
        if len(prices) < depth * 2:
            return []

        highs = prices[:, 2]  # High prices
        lows = prices[:, 3]   # Low prices

        swing_points = []
        last_swing = None
        last_price = None

        for i in range(depth, len(prices) - depth):
            # Check for swing high
            window_highs = highs[i-depth:i+depth+1]
            current_high = highs[i]

            if current_high == np.max(window_highs):
                if last_swing != 'high':
                    if last_price is None or abs(current_high - last_price) / last_price * 100 >= deviation:
                        swing_points.append({
                            'index': i,
                            'price': float(current_high),
                            'type': 'high'
                        })
                        last_swing = 'high'
                        last_price = current_high

            # Check for swing low
            window_lows = lows[i-depth:i+depth+1]
            current_low = lows[i]

            if current_low == np.min(window_lows):
                if last_swing != 'low':
                    if last_price is None or abs(current_low - last_price) / last_price * 100 >= deviation:
                        swing_points.append({
                            'index': i,
                            'price': float(current_low),
                            'type': 'low'
                        })
                        last_swing = 'low'
                        last_price = current_low

        return swing_points


class FibonacciCalculator:
    """
    Otomatik Fibonacci Retracement hesaplayıcı
    """

    @staticmethod
    def calculate_levels(swing_low: float, swing_high: float, direction: str = 'bullish') -> Dict:
        """
        Fibonacci seviyeleri hesapla

        Args:
            swing_low: En düşük nokta
            swing_high: En yüksek nokta
            direction: 'bullish' (yukarı) veya 'bearish' (aşağı)

        Returns:
            Dictionary of Fibonacci levels
        """
        range_price = swing_high - swing_low

        if direction == 'bullish':
            # Aşağıdan yukarı merdiven
            levels = {
                f"fib_{int(level*100)}": swing_low + (range_price * level)
                for level in FIBONACCI_LEVELS
            }
        else:
            # Yukarıdan aşağı merdiven
            levels = {
                f"fib_{int(level*100)}": swing_high - (range_price * level)
                for level in FIBONACCI_LEVELS
            }

        return {
            'levels': levels,
            'swing_low': swing_low,
            'swing_high': swing_high,
            'range': range_price,
            'direction': direction
        }


class MABottomHunter:
    """
    MA 7-25-99 Bottom Hunter Sistemi
    En altta olan MA'yı tespit eder ve crossover proximity'yi hesaplar
    """

    @staticmethod
    def calculate_mas(closes: np.ndarray) -> Dict:
        """
        MA 7, 25, 99 hesapla
        """
        if len(closes) < 99:
            return None

        ma7 = np.mean(closes[-7:])
        ma25 = np.mean(closes[-25:])
        ma99 = np.mean(closes[-99:])

        return {
            'ma7': float(ma7),
            'ma25': float(ma25),
            'ma99': float(ma99)
        }

    @staticmethod
    def analyze_bottom_pattern(mas: Dict, current_price: float) -> Dict:
        """
        MA Bottom Pattern analizi

        Returns:
            {
                'bottom_ma': MA adı (MA7/MA25/MA99),
                'score': 0-120 arası skor,
                'crossover_imminent': bool,
                'distance_to_next': %,
                'signal': 'STRONG_BUY'/'BUY'/'NEUTRAL'/'SELL',
                'confidence': 0-100%
            }
        """
        ma7 = mas['ma7']
        ma25 = mas['ma25']
        ma99 = mas['ma99']

        # MA'ları sırala
        sorted_mas = sorted([
            ('MA7', ma7),
            ('MA25', ma25),
            ('MA99', ma99)
        ], key=lambda x: x[1])

        bottom_ma = sorted_mas[0][0]
        bottom_value = sorted_mas[0][1]
        next_ma_value = sorted_mas[1][1]

        # Score hesapla
        score = 0

        # 1. MA7 en alttaysa = en güçlü sinyal
        if bottom_ma == 'MA7':
            score = 100

            # Crossover proximity
            distance_to_next = abs(ma7 - ma25) / current_price * 100
            if distance_to_next < 0.5:  # %0.5'den yakınsa
                score += 20  # Bonus: Crossover yakın

            # MA7 > MA25 > MA99 alignment (Golden Cross setup)
            if ma7 < ma25 < ma99:
                score += 10  # Bonus: Perfect bottom alignment

        # 2. MA25 en alttaysa = orta güçlü sinyal
        elif bottom_ma == 'MA25':
            score = 70
            distance_to_next = abs(ma25 - ma99) / current_price * 100
            if distance_to_next < 1.0:
                score += 15

        # 3. MA99 en alttaysa = zayıf sinyal (uzun vadeli destek)
        else:
            score = 40
            distance_to_next = abs(ma99 - ma25) / current_price * 100

        # Crossover imminent check
        crossover_imminent = distance_to_next < 0.5

        # Signal belirleme
        if score >= 100:
            signal = 'STRONG_BUY'
            confidence = min(95, score)
        elif score >= 70:
            signal = 'BUY'
            confidence = min(80, score)
        elif score >= 50:
            signal = 'NEUTRAL'
            confidence = 60
        else:
            signal = 'SELL'
            confidence = 50

        return {
            'bottom_ma': bottom_ma,
            'bottom_value': float(bottom_value),
            'score': score,
            'crossover_imminent': crossover_imminent,
            'distance_to_next': float(distance_to_next),
            'signal': signal,
            'confidence': confidence,
            'ma_alignment': f"{sorted_mas[0][0]} < {sorted_mas[1][0]} < {sorted_mas[2][0]}"
        }


class GoldenCrossDetector:
    """
    Golden Cross / Death Cross Detection & Retest Analysis
    - MA7 x MA25 crossover detection (4h timeframe)
    - MA25 x MA99 crossover detection (daily timeframe)
    - Retest pattern recognition (price touching MAs after cross)
    - Approaching crossover detection
    - Mathematical scoring system (0-100)
    """

    @staticmethod
    def calculate_ma_crossover(
        closes: np.ndarray,
        ma_short_period: int,
        ma_long_period: int,
        lookback: int = 20
    ) -> Dict:
        """
        MA crossover tespiti ve analizi

        Args:
            closes: Close fiyat dizisi
            ma_short_period: Kısa MA periyodu (örn: 7)
            ma_long_period: Uzun MA periyodu (örn: 25)
            lookback: Geriye dönük kaç mum kontrol edilecek

        Returns:
            {
                'crossover_detected': bool,
                'crossover_type': 'golden' | 'death' | None,
                'crossover_index': int,  # Kaç mum önce gerçekleşti
                'retest_detected': bool,
                'retest_touches': int,
                'distance_to_cross': float,  # % olarak crossover'a yakınlık
                'ma_short': float,
                'ma_long': float,
                'score': int  # 0-100 Golden Cross kalite skoru
            }
        """
        if len(closes) < max(ma_short_period, ma_long_period) + lookback:
            return None

        # MA dizilerini hesapla
        ma_short_values = []
        ma_long_values = []

        for i in range(len(closes)):
            if i >= ma_short_period - 1:
                ma_short = np.mean(closes[i - ma_short_period + 1 : i + 1])
                ma_short_values.append(ma_short)
            else:
                ma_short_values.append(None)

            if i >= ma_long_period - 1:
                ma_long = np.mean(closes[i - ma_long_period + 1 : i + 1])
                ma_long_values.append(ma_long)
            else:
                ma_long_values.append(None)

        # Son MA değerleri
        current_ma_short = ma_short_values[-1]
        current_ma_long = ma_long_values[-1]
        current_price = float(closes[-1])

        if current_ma_short is None or current_ma_long is None:
            return None

        # Crossover detection (son lookback mum içinde)
        crossover_detected = False
        crossover_type = None
        crossover_index = -1

        for i in range(len(ma_short_values) - lookback, len(ma_short_values) - 1):
            if ma_short_values[i] is None or ma_long_values[i] is None:
                continue
            if ma_short_values[i+1] is None or ma_long_values[i+1] is None:
                continue

            # Golden Cross: MA kısa, MA uzun'u aşağıdan yukarı keserse
            if ma_short_values[i] <= ma_long_values[i] and ma_short_values[i+1] > ma_long_values[i+1]:
                crossover_detected = True
                crossover_type = 'golden'
                crossover_index = len(ma_short_values) - 1 - i
                break

            # Death Cross: MA kısa, MA uzun'u yukarıdan aşağı keserse
            if ma_short_values[i] >= ma_long_values[i] and ma_short_values[i+1] < ma_long_values[i+1]:
                crossover_detected = True
                crossover_type = 'death'
                crossover_index = len(ma_short_values) - 1 - i
                break

        # Retest detection (crossover sonrası fiyat MA'lara geri dönüş yaptı mı?)
        retest_detected = False
        retest_touches = 0

        if crossover_detected and crossover_index > 0:
            # Crossover'dan sonraki mumları kontrol et
            for i in range(len(closes) - crossover_index, len(closes)):
                low = float(closes[i]) * 0.998  # Close yerine low kullanmalıyız ama basitleştirme için close
                high = float(closes[i]) * 1.002

                ma_short_at_i = ma_short_values[i] if i < len(ma_short_values) else None
                ma_long_at_i = ma_long_values[i] if i < len(ma_long_values) else None

                if ma_short_at_i is None or ma_long_at_i is None:
                    continue

                # Golden Cross sonrası: Fiyat MA7 veya MA25'e dokundu mu?
                if crossover_type == 'golden':
                    if low <= ma_short_at_i <= high or low <= ma_long_at_i <= high:
                        retest_detected = True
                        retest_touches += 1

                # Death Cross sonrası: Fiyat MA7 veya MA25'e dokundu mu?
                elif crossover_type == 'death':
                    if low <= ma_short_at_i <= high or low <= ma_long_at_i <= high:
                        retest_detected = True
                        retest_touches += 1

        # Distance to crossover (% olarak ne kadar yakın)
        distance_to_cross = abs(current_ma_short - current_ma_long) / current_price * 100

        # SCORING SYSTEM (0-100)
        score = 0

        if crossover_detected:
            # Base score: Crossover tipi
            if crossover_type == 'golden':
                score = 50  # Golden Cross base
            else:
                score = 30  # Death Cross (ters sinyal olduğu için daha düşük)

            # Bonus 1: Crossover ne kadar yeni? (0-20 puan)
            if crossover_index <= 3:
                score += 20  # Çok taze (0-3 mum)
            elif crossover_index <= 7:
                score += 15  # Taze (4-7 mum)
            elif crossover_index <= 15:
                score += 10  # Orta (8-15 mum)
            else:
                score += 5   # Eski (16+ mum)

            # Bonus 2: Retest var mı? (0-15 puan)
            if retest_detected:
                if retest_touches >= 2:
                    score += 15  # Çoklu retest = çok güçlü
                else:
                    score += 10  # Tek retest

            # Bonus 3: MA'lar ne kadar açık? (0-15 puan)
            # Daha geniş açılım = daha güçlü trend
            ma_separation = abs(current_ma_short - current_ma_long) / current_price * 100
            if ma_separation >= 2.0:
                score += 15  # %2+ açılım
            elif ma_separation >= 1.0:
                score += 10  # %1-2 açılım
            elif ma_separation >= 0.5:
                score += 5   # %0.5-1 açılım

        else:
            # Crossover yok ama yaklaşıyor mu?
            if distance_to_cross < 0.3:  # %0.3'den yakınsa
                score = 40  # Yaklaşan crossover
            elif distance_to_cross < 0.5:
                score = 30
            elif distance_to_cross < 1.0:
                score = 20
            else:
                score = 10

        return {
            'crossover_detected': crossover_detected,
            'crossover_type': crossover_type,
            'crossover_candles_ago': crossover_index if crossover_detected else None,
            'retest_detected': retest_detected,
            'retest_touches': retest_touches,
            'distance_to_cross_pct': float(distance_to_cross),
            'ma_short': float(current_ma_short),
            'ma_long': float(current_ma_long),
            'ma_short_period': ma_short_period,
            'ma_long_period': ma_long_period,
            'score': min(100, score),  # Max 100
            'current_price': current_price
        }

    @staticmethod
    def analyze_multi_timeframe_golden_cross(
        symbol: str,
        timeframes: List[str] = ['4h', '1d']
    ) -> Dict:
        """
        Çoklu timeframe Golden Cross analizi

        Args:
            symbol: Trading pair
            timeframes: Analiz edilecek timeframe'ler

        Returns:
            Multi-timeframe Golden Cross raporu
        """
        results = {}

        for tf in timeframes:
            # Binance klines çek
            klines = fetch_binance_klines(symbol, tf, limit=300)
            if klines is None:
                continue

            closes = klines[:, 4]

            # MA7 x MA25 crossover
            cross_7_25 = GoldenCrossDetector.calculate_ma_crossover(
                closes, 7, 25, lookback=20
            )

            # MA25 x MA99 crossover
            cross_25_99 = GoldenCrossDetector.calculate_ma_crossover(
                closes, 25, 99, lookback=20
            )

            results[tf] = {
                'ma7_x_ma25': cross_7_25,
                'ma25_x_ma99': cross_25_99,
                'overall_score': 0,
                'signal': 'NEUTRAL'
            }

            # Overall score ve signal hesapla
            if cross_7_25 and cross_25_99:
                # İki crossover'ın weighted average'ı
                # MA7x25 daha önemli (kısa vadeli), %60 ağırlık
                overall_score = int(cross_7_25['score'] * 0.6 + cross_25_99['score'] * 0.4)
                results[tf]['overall_score'] = overall_score

                # Signal belirleme
                if overall_score >= 80:
                    results[tf]['signal'] = 'STRONG_BUY'
                elif overall_score >= 60:
                    results[tf]['signal'] = 'BUY'
                elif overall_score >= 40:
                    results[tf]['signal'] = 'NEUTRAL'
                elif overall_score >= 20:
                    results[tf]['signal'] = 'SELL'
                else:
                    results[tf]['signal'] = 'STRONG_SELL'

        # Best timeframe seç (en yüksek score)
        best_tf = None
        best_score = 0
        for tf, data in results.items():
            if data['overall_score'] > best_score:
                best_score = data['overall_score']
                best_tf = tf

        return {
            'symbol': symbol,
            'timeframes': results,
            'best_timeframe': best_tf,
            'best_score': best_score,
            'timestamp': datetime.now().isoformat()
        }


class MultiTimeframeConfluence:
    """
    Multi-timeframe Fibonacci confluence detector
    3+ timeframe'de çakışan seviyeleri bulur ve güç skoru hesaplar
    """

    @staticmethod
    def find_confluence_zones(fib_data_list: List[Dict], tolerance: float = CONFLUENCE_TOLERANCE) -> List[Dict]:
        """
        Birden fazla timeframe'den gelen Fibonacci seviyelerini analiz eder
        ve çakışan (confluence) zonları bulur

        Args:
            fib_data_list: Her timeframe için Fibonacci data
            tolerance: Seviye eşleşme toleransı (%)

        Returns:
            List of confluence zones with power scores
        """
        all_levels = []

        # Tüm seviyeleri topla
        for i, fib_data in enumerate(fib_data_list):
            timeframe = fib_data.get('timeframe', f'TF{i}')
            for level_name, level_price in fib_data['levels'].items():
                all_levels.append({
                    'price': level_price,
                    'level': level_name,
                    'timeframe': timeframe
                })

        # Confluence zonları bul
        confluence_zones = []
        processed_prices = set()

        for level in all_levels:
            price = level['price']

            # Skip if already processed
            price_key = round(price, 2)
            if price_key in processed_prices:
                continue

            # Bu fiyata yakın tüm seviyeleri bul
            matches = []
            for other_level in all_levels:
                other_price = other_level['price']
                if abs(other_price - price) / price <= tolerance:
                    matches.append(other_level)

            # En az 2 timeframe çakışması varsa confluence zone
            if len(matches) >= 2:
                unique_timeframes = len(set([m['timeframe'] for m in matches]))

                # Power score: Çakışan timeframe sayısı * 30
                power_score = unique_timeframes * 30

                avg_price = np.mean([m['price'] for m in matches])

                confluence_zones.append({
                    'price': float(avg_price),
                    'power_score': power_score,
                    'confluence_count': len(matches),
                    'timeframes': unique_timeframes,
                    'levels': [f"{m['timeframe']}:{m['level']}" for m in matches]
                })

                processed_prices.add(price_key)

        # Power score'a göre sırala
        confluence_zones.sort(key=lambda x: x['power_score'], reverse=True)

        return confluence_zones


def fetch_binance_klines(symbol: str, interval: str, limit: int = 500) -> Optional[np.ndarray]:
    """
    Binance'den kline (OHLC) verisi çek

    Returns:
        numpy array of [timestamp, open, high, low, close, volume]
    """
    try:
        url = f"{BINANCE_API}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            if response.status_code == 418:
                logger.error(f"Binance API rate limit (418) - IP banned temporarily")
            else:
                logger.error(f"Binance API error: {response.status_code}")
            return None

        klines = response.json()

        # Convert to numpy array [open, high, low, close]
        data = np.array([[
            float(k[0]),  # timestamp
            float(k[1]),  # open
            float(k[2]),  # high
            float(k[3]),  # low
            float(k[4]),  # close
            float(k[5])   # volume
        ] for k in klines])

        return data

    except Exception as e:
        logger.error(f"Error fetching klines: {str(e)}")
        return None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Quantum Ladder Strategy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/analyze', methods=['POST'])
def analyze_quantum_ladder():
    """
    Quantum Ladder analizi - Ana endpoint

    POST body:
    {
        "symbol": "BTCUSDT",
        "timeframes": ["15m", "1h", "4h"]  # Opsiyonel, default: ["15m", "1h", "4h"]
    }
    """
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'BTCUSDT')
        timeframes = data.get('timeframes', ['15m', '1h', '4h'])
        limit = data.get('limit', 500)

        logger.info(f"Analyzing {symbol} on timeframes: {timeframes} with {limit} candles")

        # Multi-timeframe analiz
        all_fib_data = []
        primary_timeframe_data = None

        for timeframe in timeframes:
            # Kline verisi çek
            klines = fetch_binance_klines(symbol, timeframe, limit)
            if klines is None:
                continue

            # ZigZag hesapla
            zigzag = ZigZagCalculator()
            swing_points = zigzag.calculate(klines)

            if len(swing_points) < 2:
                continue

            # Son 2 swing point'i al
            recent_swings = swing_points[-2:]

            # Fibonacci seviyeleri hesapla
            if recent_swings[0]['type'] == 'low' and recent_swings[1]['type'] == 'high':
                # Bullish (aşağıdan yukarı)
                fib_calc = FibonacciCalculator()
                fib_data = fib_calc.calculate_levels(
                    recent_swings[0]['price'],
                    recent_swings[1]['price'],
                    'bullish'
                )
                fib_data['timeframe'] = timeframe
                fib_data['swing_points'] = recent_swings
                all_fib_data.append(fib_data)

            elif recent_swings[0]['type'] == 'high' and recent_swings[1]['type'] == 'low':
                # Bearish (yukarıdan aşağı)
                fib_calc = FibonacciCalculator()
                fib_data = fib_calc.calculate_levels(
                    recent_swings[1]['price'],
                    recent_swings[0]['price'],
                    'bearish'
                )
                fib_data['timeframe'] = timeframe
                fib_data['swing_points'] = recent_swings
                all_fib_data.append(fib_data)

            # İlk timeframe'i primary olarak sakla
            if primary_timeframe_data is None:
                primary_timeframe_data = {
                    'klines': klines,
                    'timeframe': timeframe,
                    'swing_points': swing_points
                }

        if not all_fib_data:
            return jsonify({
                'success': False,
                'error': 'Binance API geçici olarak sınırlandı. Lütfen birkaç dakika sonra tekrar deneyin.'
            }), 429  # Use 429 for rate limiting instead of 400

        # MA Bottom Hunter analizi (primary timeframe'de)
        ma_analysis = None
        if primary_timeframe_data:
            closes = primary_timeframe_data['klines'][:, 4]
            current_price = float(closes[-1])

            ma_hunter = MABottomHunter()
            mas = ma_hunter.calculate_mas(closes)

            if mas:
                ma_analysis = ma_hunter.analyze_bottom_pattern(mas, current_price)
                ma_analysis['mas'] = mas

        # Golden Cross analizi (tüm timeframe'lerde)
        golden_cross_detector = GoldenCrossDetector()
        golden_cross_analysis = golden_cross_detector.analyze_multi_timeframe_golden_cross(
            symbol,
            timeframes=['4h', '1d'] + timeframes  # 4h ve daily'yi de ekle
        )

        # Multi-timeframe confluence hesapla
        confluence_detector = MultiTimeframeConfluence()
        confluence_zones = confluence_detector.find_confluence_zones(all_fib_data)

        # En yakın destek/direnç seviyelerini bul
        nearest_support = None
        nearest_resistance = None

        for zone in confluence_zones:
            if zone['price'] < current_price:
                if nearest_support is None or zone['price'] > nearest_support['price']:
                    nearest_support = zone
            elif zone['price'] > current_price:
                if nearest_resistance is None or zone['price'] < nearest_resistance['price']:
                    nearest_resistance = zone

        # Final sinyal oluştur
        final_signal = 'NEUTRAL'
        final_confidence = 50

        if ma_analysis:
            # MA analizi ile Fibonacci confluenc'i birleştir
            ma_signal = ma_analysis['signal']
            ma_confidence = ma_analysis['confidence']

            # Confluence gücü ile weighted average
            if confluence_zones:
                max_confluence_power = confluence_zones[0]['power_score']
                confluence_weight = min(100, max_confluence_power) / 100

                # Ağırlıklı ortalama
                final_confidence = int(ma_confidence * 0.6 + max_confluence_power * 0.4)
            else:
                final_confidence = ma_confidence

            final_signal = ma_signal

        # Response oluştur
        return jsonify({
            'success': True,
            'data': {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),

                # MA Bottom Hunter sonuçları
                'ma_bottom_hunter': ma_analysis,

                # Golden Cross analizi (YENI!)
                'golden_cross': golden_cross_analysis,

                # Fibonacci ladder structure
                'fibonacci_ladders': [
                    {
                        'timeframe': fib['timeframe'],
                        'direction': fib['direction'],
                        'swing_low': fib['swing_low'],
                        'swing_high': fib['swing_high'],
                        'range': fib['range'],
                        'levels': fib['levels']
                    }
                    for fib in all_fib_data
                ],

                # Multi-timeframe confluence zones
                'confluence_zones': confluence_zones[:10],  # Top 10

                # Nearest levels
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,

                # Final signal
                'signal': final_signal,
                'confidence': final_confidence,

                # Metadata
                'timeframes_analyzed': len(all_fib_data),
                'total_confluence_zones': len(confluence_zones)
            }
        })

    except Exception as e:
        logger.error(f"Error in quantum ladder analysis: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/quick-scan', methods=['GET'])
def quick_scan():
    """
    Hızlı çoklu coin taraması
    Query params: ?symbols=BTCUSDT,ETHUSDT,SOLUSDT
    """
    try:
        symbols_param = request.args.get('symbols', 'BTCUSDT,ETHUSDT')
        symbols = [s.strip() for s in symbols_param.split(',')]

        results = []

        for symbol in symbols[:10]:  # Max 10 coin
            try:
                # Quick analysis (sadece 1h timeframe)
                klines = fetch_binance_klines(symbol, '1h', limit=200)
                if klines is None:
                    continue

                closes = klines[:, 4]
                current_price = float(closes[-1])

                # MA Bottom Hunter
                ma_hunter = MABottomHunter()
                mas = ma_hunter.calculate_mas(closes)

                if mas:
                    ma_analysis = ma_hunter.analyze_bottom_pattern(mas, current_price)

                    results.append({
                        'symbol': symbol,
                        'price': current_price,
                        'signal': ma_analysis['signal'],
                        'confidence': ma_analysis['confidence'],
                        'bottom_ma': ma_analysis['bottom_ma'],
                        'score': ma_analysis['score']
                    })
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                continue

        # Confidence'a göre sırala
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return jsonify({
            'success': True,
            'data': {
                'scanned_count': len(results),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error in quick scan: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    logger.info("🚀 Quantum Ladder Service başlatılıyor...")
    logger.info("Port: 5022")
    app.run(host='0.0.0.0', port=5022, debug=False)
