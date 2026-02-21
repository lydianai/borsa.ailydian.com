"""
MULTI-TIMEFRAME CONFLUENCE ANALYZER
===================================

Advanced multi-timeframe analysis for crypto trading including:
- Timeframe harmony scoring
- Confluence detection across multiple timeframes
- Signal strength assessment
- Divergence identification
- Trend alignment analysis

Features:
- 9 timeframe analysis (1m to 1M)
- Harmonic pattern detection
- Confluence scoring system
- Divergence analysis
- Momentum convergence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import defaultdict
import math

class Timeframe(Enum):
    """Supported timeframes"""
    ONE_MINUTE = ("1m", 1)
    FIVE_MINUTE = ("5m", 5)
    FIFTEEN_MINUTE = ("15m", 15)
    THIRTY_MINUTE = ("30m", 30)
    ONE_HOUR = ("1h", 60)
    TWO_HOUR = ("2h", 120)
    FOUR_HOUR = ("4h", 240)
    SIX_HOUR = ("6h", 360)
    TWELVE_HOUR = ("12h", 720)
    ONE_DAY = ("1d", 1440)
    THREE_DAY = ("3d", 4320)
    ONE_WEEK = ("1w", 10080)
    ONE_MONTH = ("1M", 43200)  # Approximate

class SignalType(Enum):
    """Signal types for confluence analysis"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    DIVERGENCE = "divergence"
    CONVERGENCE = "convergence"
    BREAKOUT = "breakout"
    RETEST = "retest"

class ConfluenceLevel(Enum):
    """Confluence strength levels"""
    VERY_WEAK = "very_weak"      # 1 timeframe
    WEAK = "weak"                # 2 timeframes
    MODERATE = "moderate"        # 3 timeframes
    STRONG = "strong"            # 4-5 timeframes
    VERY_STRONG = "very_strong"  # 6+ timeframes

@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe"""
    timeframe: Timeframe
    signal_type: SignalType
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    price_level: float
    indicator_values: Dict[str, float]
    timestamp: pd.Timestamp
    metadata: Dict[str, Any]

@dataclass
class ConfluenceZone:
    """Area of confluence across multiple timeframes"""
    price_level: float
    confluence_score: float  # 0.0 - 10.0
    confluence_level: ConfluenceLevel
    timeframe_signals: List[TimeframeSignal]
    support_resistance: str  # 'support', 'resistance', 'neutral'
    harmonic_alignment: bool  # True if harmonic pattern detected
    divergence_present: bool  # True if divergence detected
    timestamp: pd.Timestamp

@dataclass
class MultiTimeframeAnalysis:
    """Complete multi-timeframe analysis"""
    symbol: str
    primary_timeframe: Timeframe
    confluence_zones: List[ConfluenceZone]
    dominant_trend: SignalType
    trend_strength: float  # 0.0 - 1.0
    momentum_alignment: float  # -1.0 to 1.0 (bearish to bullish)
    harmonic_patterns: List[Dict[str, Any]]
    divergences: List[Dict[str, Any]]
    key_levels: List[Dict[str, Any]]
    timestamp: pd.Timestamp

class MultiTimeframeAnalyzer:
    """
    Advanced Multi-Timeframe Confluence Analyzer
    
    Analyzes multiple timeframes simultaneously to:
    1. Identify confluence zones
    2. Assess signal strength across timeframes
    3. Detect harmonic patterns
    4. Identify divergences
    5. Evaluate trend alignment
    6. Generate confluence scores
    """
    
    def __init__(
        self,
        symbol: str,
        primary_timeframe: Timeframe = Timeframe.ONE_HOUR,
        timeframes_to_analyze: List[Timeframe] = None,
        confluence_threshold: float = 3.0,  # Minimum confluence score
        harmonic_sensitivity: float = 0.7,  # Harmonic pattern sensitivity
        divergence_threshold: float = 0.02   # 2% minimum divergence
    ):
        self.symbol = symbol
        self.primary_timeframe = primary_timeframe
        self.timeframes_to_analyze = timeframes_to_analyze or [
            Timeframe.FIFTEEN_MINUTE,
            Timeframe.ONE_HOUR,
            Timeframe.FOUR_HOUR,
            Timeframe.ONE_DAY,
            Timeframe.ONE_WEEK
        ]
        self.confluence_threshold = confluence_threshold
        self.harmonic_sensitivity = harmonic_sensitivity
        self.divergence_threshold = divergence_threshold
        
        # Storage for analysis
        self.timeframe_data: Dict[Timeframe, pd.DataFrame] = {}
        self.timeframe_signals: Dict[Timeframe, List[TimeframeSignal]] = {}
        self.analysis_history: List[MultiTimeframeAnalysis] = []
        
        # Cached results
        self._cached_confluence_zones: Optional[List[ConfluenceZone]] = None
        self._cache_timestamp: Optional[pd.Timestamp] = None
    
    def add_timeframe_data(
        self,
        timeframe: Timeframe,
        data: pd.DataFrame
    ):
        """
        Add OHLCV data for a specific timeframe
        
        Args:
            timeframe: Timeframe enum
            data: DataFrame with OHLCV data indexed by timestamp
        """
        self.timeframe_data[timeframe] = data.copy()
        # Invalidate cache when new data is added
        self._cached_confluence_zones = None
        self._cache_timestamp = None
    
    def analyze_all_timeframes(self) -> MultiTimeframeAnalysis:
        """
        Perform complete multi-timeframe analysis
        
        Returns:
            MultiTimeframeAnalysis object with all results
        """
        # Generate signals for each timeframe
        for timeframe in self.timeframes_to_analyze:
            if timeframe in self.timeframe_data:
                signals = self._generate_timeframe_signals(timeframe)
                self.timeframe_signals[timeframe] = signals
        
        # Identify confluence zones
        confluence_zones = self._identify_confluence_zones()
        
        # Analyze dominant trend
        dominant_trend, trend_strength = self._analyze_dominant_trend()
        
        # Calculate momentum alignment
        momentum_alignment = self._calculate_momentum_alignment()
        
        # Detect harmonic patterns
        harmonic_patterns = self._detect_harmonic_patterns()
        
        # Identify divergences
        divergences = self._identify_divergences()
        
        # Extract key levels
        key_levels = self._extract_key_levels(confluence_zones)
        
        # Create analysis object
        analysis = MultiTimeframeAnalysis(
            symbol=self.symbol,
            primary_timeframe=self.primary_timeframe,
            confluence_zones=confluence_zones,
            dominant_trend=dominant_trend,
            trend_strength=trend_strength,
            momentum_alignment=momentum_alignment,
            harmonic_patterns=harmonic_patterns,
            divergences=divergences,
            key_levels=key_levels,
            timestamp=pd.Timestamp.now()
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _generate_timeframe_signals(
        self,
        timeframe: Timeframe
    ) -> List[TimeframeSignal]:
        """
        Generate signals for a specific timeframe
        
        Args:
            timeframe: Timeframe to analyze
            
        Returns:
            List of TimeframeSignal objects
        """
        if timeframe not in self.timeframe_data:
            return []
        
        df = self.timeframe_data[timeframe]
        signals = []
        
        if len(df) < 50:  # Need sufficient data
            return signals
        
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Generate signals based on indicators
        for i in range(20, len(df)):  # Start after enough data
            signal = self._generate_single_signal(df, i, timeframe)
            if signal:
                signals.append(signal)
        
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for signal generation
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd_line'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
    
    def _generate_single_signal(
        self,
        df: pd.DataFrame,
        index: int,
        timeframe: Timeframe
    ) -> Optional[TimeframeSignal]:
        """
        Generate a single signal at a specific index
        
        Args:
            df: DataFrame with indicators
            index: Index to generate signal for
            timeframe: Timeframe enum
            
        Returns:
            TimeframeSignal object or None
        """
        if index < 20 or index >= len(df):
            return None
        
        # Get current values
        current_row = df.iloc[index]
        prev_row = df.iloc[index - 1]
        
        # Calculate signal components
        ma_bullish = current_row['sma_20'] > current_row['sma_50'] and prev_row['sma_20'] <= prev_row['sma_50']
        ma_bearish = current_row['sma_20'] < current_row['sma_50'] and prev_row['sma_20'] >= prev_row['sma_50']
        
        macd_bullish = current_row['macd_histogram'] > 0 and prev_row['macd_histogram'] <= 0
        macd_bearish = current_row['macd_histogram'] < 0 and prev_row['macd_histogram'] >= 0
        
        rsi_oversold = current_row['rsi'] < 30 and prev_row['rsi'] >= 30
        rsi_overbought = current_row['rsi'] > 70 and prev_row['rsi'] <= 70
        
        stoch_bullish = current_row['stoch_k'] > current_row['stoch_d'] and prev_row['stoch_k'] <= prev_row['stoch_d']
        stoch_bearish = current_row['stoch_k'] < current_row['stoch_d'] and prev_row['stoch_k'] >= prev_row['stoch_d']
        
        # Determine signal type
        bullish_signals = sum([ma_bullish, macd_bullish, rsi_oversold, stoch_bullish])
        bearish_signals = sum([ma_bearish, macd_bearish, rsi_overbought, stoch_bearish])
        
        if bullish_signals >= 2 and bullish_signals > bearish_signals:
            signal_type = SignalType.BULLISH
            strength = min(1.0, bullish_signals / 4.0)
        elif bearish_signals >= 2 and bearish_signals > bullish_signals:
            signal_type = SignalType.BEARISH
            strength = min(1.0, bearish_signals / 4.0)
        else:
            signal_type = SignalType.NEUTRAL
            strength = 0.5
        
        # Calculate confidence based on indicator alignment
        total_signals = bullish_signals + bearish_signals
        confidence = min(1.0, total_signals / 4.0) if total_signals > 0 else 0.0
        
        # Price level (use current close)
        price_level = current_row['close']
        
        # Indicator values
        indicator_values = {
            'sma_20': current_row['sma_20'],
            'sma_50': current_row['sma_50'],
            'macd_histogram': current_row['macd_histogram'],
            'rsi': current_row['rsi'],
            'stoch_k': current_row['stoch_k'],
            'stoch_d': current_row['stoch_d']
        }
        
        return TimeframeSignal(
            timeframe=timeframe,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price_level=price_level,
            indicator_values=indicator_values,
            timestamp=current_row.name,
            metadata={
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'total_signals': total_signals
            }
        )
    
    def _identify_confluence_zones(self) -> List[ConfluenceZone]:
        """
        Identify areas of confluence across multiple timeframes
        
        Returns:
            List of ConfluenceZone objects
        """
        # Use cached results if available and recent
        if (self._cached_confluence_zones is not None and 
            self._cache_timestamp is not None and
            (pd.Timestamp.now() - self._cache_timestamp).seconds < 60):  # Cache for 1 minute
            return self._cached_confluence_zones
        
        confluence_zones = []
        
        # Get all price levels from all timeframes
        all_price_levels = []
        timeframe_signals_combined = []
        
        for timeframe, signals in self.timeframe_signals.items():
            for signal in signals[-50:]:  # Last 50 signals per timeframe
                all_price_levels.append(signal.price_level)
                timeframe_signals_combined.append(signal)
        
        if not all_price_levels:
            self._cached_confluence_zones = confluence_zones
            self._cache_timestamp = pd.Timestamp.now()
            return confluence_zones
        
        # Cluster price levels
        clustered_levels = self._cluster_price_levels(all_price_levels)
        
        # Create confluence zones
        for price_level in clustered_levels:
            zone_signals = self._find_signals_near_level(price_level, timeframe_signals_combined)
            
            if zone_signals:
                # Calculate confluence score
                confluence_score = self._calculate_confluence_score(zone_signals)
                
                # Determine confluence level
                confluence_level = self._determine_confluence_level(len(zone_signals))
                
                # Determine support/resistance
                support_resistance = self._determine_support_resistance(zone_signals)
                
                # Check for harmonic alignment
                harmonic_alignment = self._check_harmonic_alignment(zone_signals)
                
                # Check for divergence
                divergence_present = self._check_divergence_in_zone(zone_signals)
                
                confluence_zone = ConfluenceZone(
                    price_level=price_level,
                    confluence_score=confluence_score,
                    confluence_level=confluence_level,
                    timeframe_signals=zone_signals,
                    support_resistance=support_resistance,
                    harmonic_alignment=harmonic_alignment,
                    divergence_present=divergence_present,
                    timestamp=pd.Timestamp.now()
                )
                
                # Only include zones above threshold
                if confluence_score >= self.confluence_threshold:
                    confluence_zones.append(confluence_zone)
        
        # Sort by confluence score
        confluence_zones.sort(key=lambda x: x.confluence_score, reverse=True)
        
        # Cache results
        self._cached_confluence_zones = confluence_zones
        self._cache_timestamp = pd.Timestamp.now()
        
        return confluence_zones
    
    def _cluster_price_levels(
        self,
        price_levels: List[float],
        cluster_radius: float = 0.005  # 0.5% cluster radius
    ) -> List[float]:
        """
        Cluster similar price levels
        
        Args:
            price_levels: List of price levels
            cluster_radius: Percentage radius for clustering
            
        Returns:
            List of clustered price levels
        """
        if not price_levels:
            return []
        
        # Sort price levels
        sorted_levels = sorted(price_levels)
        clusters = []
        
        # Cluster levels within radius
        current_cluster = [sorted_levels[0]]
        cluster_center = sorted_levels[0]
        
        for level in sorted_levels[1:]:
            # Check if level is within cluster radius
            radius = cluster_center * cluster_radius
            
            if abs(level - cluster_center) <= radius:
                current_cluster.append(level)
                # Update cluster center
                cluster_center = np.mean(current_cluster)
            else:
                # Start new cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
                cluster_center = level
        
        # Add last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return clusters
    
    def _find_signals_near_level(
        self,
        price_level: float,
        signals: List[TimeframeSignal],
        tolerance: float = 0.005  # 0.5% tolerance
    ) -> List[TimeframeSignal]:
        """
        Find signals near a specific price level
        
        Args:
            price_level: Target price level
            signals: List of signals to search
            tolerance: Percentage tolerance for matching
            
        Returns:
            List of matching signals
        """
        matching_signals = []
        tolerance_amount = price_level * tolerance
        
        for signal in signals:
            if abs(signal.price_level - price_level) <= tolerance_amount:
                matching_signals.append(signal)
        
        return matching_signals
    
    def _calculate_confluence_score(
        self,
        signals: List[TimeframeSignal]
    ) -> float:
        """
        Calculate confluence score for a group of signals
        
        Args:
            signals: List of signals at same price level
            
        Returns:
            Confluence score (0.0 - 10.0)
        """
        if not signals:
            return 0.0
        
        # Base score: number of timeframes
        base_score = len(signals) * 1.5
        
        # Strength score: average signal strength
        avg_strength = np.mean([s.strength for s in signals])
        strength_score = avg_strength * 3.0
        
        # Confidence score: average confidence
        avg_confidence = np.mean([s.confidence for s in signals])
        confidence_score = avg_confidence * 2.0
        
        # Agreement score: percentage of signals in same direction
        bullish_count = sum(1 for s in signals if s.signal_type == SignalType.BULLISH)
        bearish_count = sum(1 for s in signals if s.signal_type == SignalType.BEARISH)
        total_directional = bullish_count + bearish_count
        
        if total_directional > 0:
            agreement = max(bullish_count, bearish_count) / total_directional
            agreement_score = agreement * 2.0
        else:
            agreement_score = 1.0  # Neutral agreement
        
        # Timeframe diversity bonus
        unique_timeframes = len(set(s.timeframe for s in signals))
        diversity_bonus = unique_timeframes * 0.5 if unique_timeframes >= 3 else 0.0
        
        # Calculate total score
        total_score = base_score + strength_score + confidence_score + agreement_score + diversity_bonus
        
        # Normalize to 0-10 scale
        max_possible = 1.5*len(signals) + 3.0 + 2.0 + 2.0 + 0.5*len(signals)
        normalized_score = min(10.0, (total_score / max_possible) * 10.0) if max_possible > 0 else 0.0
        
        return normalized_score
    
    def _determine_confluence_level(
        self,
        signal_count: int
    ) -> ConfluenceLevel:
        """
        Determine confluence level based on signal count
        
        Args:
            signal_count: Number of signals in confluence
            
        Returns:
            ConfluenceLevel enum
        """
        if signal_count >= 6:
            return ConfluenceLevel.VERY_STRONG
        elif signal_count >= 4:
            return ConfluenceLevel.STRONG
        elif signal_count >= 3:
            return ConfluenceLevel.MODERATE
        elif signal_count >= 2:
            return ConfluenceLevel.WEAK
        else:
            return ConfluenceLevel.VERY_WEAK
    
    def _determine_support_resistance(
        self,
        signals: List[TimeframeSignal]
    ) -> str:
        """
        Determine if zone is support, resistance, or neutral
        
        Args:
            signals: List of signals at zone
            
        Returns:
            'support', 'resistance', or 'neutral'
        """
        bullish_signals = [s for s in signals if s.signal_type == SignalType.BULLISH]
        bearish_signals = [s for s in signals if s.signal_type == SignalType.BEARISH]
        
        if len(bullish_signals) > len(bearish_signals) * 1.5:
            return 'support'
        elif len(bearish_signals) > len(bullish_signals) * 1.5:
            return 'resistance'
        else:
            return 'neutral'
    
    def _check_harmonic_alignment(
        self,
        signals: List[TimeframeSignal]
    ) -> bool:
        """
        Check for harmonic alignment among signals
        
        Args:
            signals: List of signals at zone
            
        Returns:
            True if harmonic alignment detected
        """
        if len(signals) < 3:
            return False
        
        # Check if signals are from different time horizons
        short_term = [s for s in signals if s.timeframe.value[1] <= 60]  # <= 1h
        medium_term = [s for s in signals if 60 < s.timeframe.value[1] <= 1440]  # 1h < x <= 1d
        long_term = [s for s in signals if s.timeframe.value[1] > 1440]  # > 1d
        
        # Harmonic if we have signals from at least 2 different time horizons
        time_horizons = sum([1 for term in [short_term, medium_term, long_term] if term])
        
        return time_horizons >= 2
    
    def _check_divergence_in_zone(
        self,
        signals: List[TimeframeSignal]
    ) -> bool:
        """
        Check for divergence among signals in zone
        
        Args:
            signals: List of signals at zone
            
        Returns:
            True if divergence detected
        """
        if len(signals) < 2:
            return False
        
        # Look for opposing signals (bullish vs bearish)
        bullish_signals = [s for s in signals if s.signal_type == SignalType.BULLISH]
        bearish_signals = [s for s in signals if s.signal_type == SignalType.BEARISH]
        
        # Divergence if we have both bullish and bearish signals
        return len(bullish_signals) > 0 and len(bearish_signals) > 0
    
    def _analyze_dominant_trend(self) -> Tuple[SignalType, float]:
        """
        Analyze dominant trend across all timeframes
        
        Returns:
            Tuple of (dominant_trend, trend_strength)
        """
        if not self.timeframe_signals:
            return SignalType.NEUTRAL, 0.0
        
        # Weight timeframes by importance (higher timeframe = higher weight)
        timeframe_weights = {
            Timeframe.ONE_MINUTE: 0.1,
            Timeframe.FIVE_MINUTE: 0.2,
            Timeframe.FIFTEEN_MINUTE: 0.3,
            Timeframe.THIRTY_MINUTE: 0.4,
            Timeframe.ONE_HOUR: 0.6,
            Timeframe.TWO_HOUR: 0.7,
            Timeframe.FOUR_HOUR: 0.8,
            Timeframe.SIX_HOUR: 0.85,
            Timeframe.TWELVE_HOUR: 0.9,
            Timeframe.ONE_DAY: 1.0,
            Timeframe.THREE_DAY: 1.1,
            Timeframe.ONE_WEEK: 1.2,
            Timeframe.ONE_MONTH: 1.3
        }
        
        bullish_weighted = 0.0
        bearish_weighted = 0.0
        total_weight = 0.0
        
        for timeframe, signals in self.timeframe_signals.items():
            if not signals:
                continue
                
            weight = timeframe_weights.get(timeframe, 0.5)
            
            # Use recent signals (last 10)
            recent_signals = signals[-10:]
            
            for signal in recent_signals:
                if signal.signal_type == SignalType.BULLISH:
                    bullish_weighted += signal.strength * signal.confidence * weight
                elif signal.signal_type == SignalType.BEARISH:
                    bearish_weighted += signal.strength * signal.confidence * weight
                
                total_weight += weight
        
        if total_weight == 0:
            return SignalType.NEUTRAL, 0.0
        
        # Normalize
        bullish_normalized = bullish_weighted / total_weight
        bearish_normalized = bearish_weighted / total_weight
        
        # Determine dominant trend
        if abs(bullish_normalized - bearish_normalized) < 0.1:
            dominant_trend = SignalType.NEUTRAL
            trend_strength = 0.0
        elif bullish_normalized > bearish_normalized:
            dominant_trend = SignalType.BULLISH
            trend_strength = min(1.0, (bullish_normalized - bearish_normalized) / 2.0)
        else:
            dominant_trend = SignalType.BEARISH
            trend_strength = min(1.0, (bearish_normalized - bullish_normalized) / 2.0)
        
        return dominant_trend, trend_strength
    
    def _calculate_momentum_alignment(self) -> float:
        """
        Calculate momentum alignment across timeframes
        
        Returns:
            Momentum alignment (-1.0 to 1.0, bearish to bullish)
        """
        if not self.timeframe_signals:
            return 0.0
        
        momentum_scores = []
        
        for timeframe, signals in self.timeframe_signals.items():
            if not signals:
                continue
            
            # Calculate average momentum for this timeframe
            recent_signals = signals[-20:]  # Last 20 signals
            if not recent_signals:
                continue
            
            bullish_momentum = sum(s.strength for s in recent_signals if s.signal_type == SignalType.BULLISH)
            bearish_momentum = sum(s.strength for s in recent_signals if s.signal_type == SignalType.BEARISH)
            
            timeframe_momentum = bullish_momentum - bearish_momentum
            
            # Weight by timeframe importance
            weight_map = {
                Timeframe.ONE_MINUTE: 0.1,
                Timeframe.FIVE_MINUTE: 0.2,
                Timeframe.FIFTEEN_MINUTE: 0.3,
                Timeframe.THIRTY_MINUTE: 0.4,
                Timeframe.ONE_HOUR: 0.6,
                Timeframe.TWO_HOUR: 0.7,
                Timeframe.FOUR_HOUR: 0.8,
                Timeframe.SIX_HOUR: 0.85,
                Timeframe.TWELVE_HOUR: 0.9,
                Timeframe.ONE_DAY: 1.0,
                Timeframe.THREE_DAY: 1.1,
                Timeframe.ONE_WEEK: 1.2,
                Timeframe.ONE_MONTH: 1.3
            }
            
            weight = weight_map.get(timeframe, 0.5)
            weighted_momentum = timeframe_momentum * weight
            momentum_scores.append(weighted_momentum)
        
        if not momentum_scores:
            return 0.0
        
        # Calculate average momentum alignment
        avg_momentum = np.mean(momentum_scores)
        
        # Normalize to -1.0 to 1.0 range
        max_expected = 5.0  # Maximum expected momentum
        normalized_momentum = max(-1.0, min(1.0, avg_momentum / max_expected))
        
        return normalized_momentum
    
    def _detect_harmonic_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect harmonic patterns across timeframes
        
        Returns:
            List of detected harmonic patterns
        """
        patterns = []
        
        # This is a simplified harmonic pattern detection
        # In practice, this would involve complex pattern recognition
        
        for timeframe, signals in self.timeframe_signals.items():
            if len(signals) < 5:  # Need at least 5 signals
                continue
            
            # Look for ABCD pattern (simplified)
            recent_signals = signals[-10:]  # Last 10 signals
            
            for i in range(len(recent_signals) - 3):
                a_signal = recent_signals[i]
                b_signal = recent_signals[i + 1]
                c_signal = recent_signals[i + 2]
                d_signal = recent_signals[i + 3]
                
                # Check for potential ABCD pattern
                if self._is_abcd_pattern(a_signal, b_signal, c_signal, d_signal):
                    pattern = {
                        'type': 'abcd',
                        'timeframe': timeframe.value[0],
                        'points': [
                            {'price': a_signal.price_level, 'timestamp': a_signal.timestamp.isoformat()},
                            {'price': b_signal.price_level, 'timestamp': b_signal.timestamp.isoformat()},
                            {'price': c_signal.price_level, 'timestamp': c_signal.timestamp.isoformat()},
                            {'price': d_signal.price_level, 'timestamp': d_signal.timestamp.isoformat()}
                        ],
                        'confidence': self._calculate_abcd_confidence(a_signal, b_signal, c_signal, d_signal),
                        'timestamp': d_signal.timestamp.isoformat()
                    }
                    patterns.append(pattern)
        
        return patterns
    
    def _is_abcd_pattern(
        self,
        a: TimeframeSignal,
        b: TimeframeSignal,
        c: TimeframeSignal,
        d: TimeframeSignal
    ) -> bool:
        """
        Check if four signals form an ABCD pattern
        
        Args:
            a, b, c, d: Four consecutive signals
            
        Returns:
            True if ABCD pattern detected
        """
        # Simplified ABCD pattern detection
        # In practice, this would involve Fibonacci ratios and precise measurements
        
        # Basic structure check
        ab_move = abs(b.price_level - a.price_level)
        bc_move = abs(c.price_level - b.price_level)
        cd_move = abs(d.price_level - c.price_level)
        
        # ABCD ratio should be approximately 1.27 - 1.618
        if bc_move > 0 and cd_move > 0:
            bc_cd_ratio = cd_move / bc_move
            return 1.27 <= bc_cd_ratio <= 1.618
        
        return False
    
    def _calculate_abcd_confidence(
        self,
        a: TimeframeSignal,
        b: TimeframeSignal,
        c: TimeframeSignal,
        d: TimeframeSignal
    ) -> float:
        """
        Calculate confidence in ABCD pattern
        
        Args:
            a, b, c, d: Four consecutive signals
            
        Returns:
            Confidence level (0.0 - 1.0)
        """
        # Weight factors
        strength_weight = 0.3
        confidence_weight = 0.3
        timeframe_weight = 0.2
        alignment_weight = 0.2
        
        # Calculate components
        avg_strength = np.mean([s.strength for s in [a, b, c, d]])
        avg_confidence = np.mean([s.confidence for s in [a, b, c, d]])
        
        # Timeframe factor (higher timeframe = higher confidence)
        timeframe_map = {
            Timeframe.ONE_MINUTE: 0.1,
            Timeframe.FIVE_MINUTE: 0.2,
            Timeframe.FIFTEEN_MINUTE: 0.3,
            Timeframe.THIRTY_MINUTE: 0.4,
            Timeframe.ONE_HOUR: 0.6,
            Timeframe.TWO_HOUR: 0.7,
            Timeframe.FOUR_HOUR: 0.8,
            Timeframe.SIX_HOUR: 0.85,
            Timeframe.TWELVE_HOUR: 0.9,
            Timeframe.ONE_DAY: 1.0,
            Timeframe.THREE_DAY: 1.1,
            Timeframe.ONE_WEEK: 1.2,
            Timeframe.ONE_MONTH: 1.3
        }
        
        timeframe_factor = np.mean([
            timeframe_map.get(s.timeframe, 0.5) for s in [a, b, c, d]
        ])
        
        # Alignment factor (how well signals align)
        bullish_count = sum(1 for s in [a, b, c, d] if s.signal_type == SignalType.BULLISH)
        bearish_count = sum(1 for s in [a, b, c, d] if s.signal_type == SignalType.BEARISH)
        alignment_factor = 1.0 - (abs(bullish_count - bearish_count) / 4.0)
        
        # Calculate weighted confidence
        confidence = (
            avg_strength * strength_weight +
            avg_confidence * confidence_weight +
            timeframe_factor * timeframe_weight +
            alignment_factor * alignment_weight
        )
        
        return min(1.0, confidence)
    
    def _identify_divergences(self) -> List[Dict[str, Any]]:
        """
        Identify divergences across timeframes
        
        Returns:
            List of detected divergences
        """
        divergences = []
        
        for timeframe, signals in self.timeframe_signals.items():
            if len(signals) < 10:  # Need sufficient data
                continue
            
            # Look for price/momentum divergences
            recent_signals = signals[-20:]  # Last 20 signals
            
            for i in range(len(recent_signals) - 5):
                window_signals = recent_signals[i:i+5]
                
                # Check for bullish divergence (price making lower lows, momentum making higher lows)
                if self._is_bullish_divergence(window_signals):
                    divergence = {
                        'type': 'bullish_divergence',
                        'timeframe': timeframe.value[0],
                        'signals': [
                            {
                                'price': s.price_level,
                                'timestamp': s.timestamp.isoformat(),
                                'signal_type': s.signal_type.value
                            }
                            for s in window_signals
                        ],
                        'confidence': self._calculate_divergence_confidence(window_signals),
                        'timestamp': window_signals[-1].timestamp.isoformat()
                    }
                    divergences.append(divergence)
                
                # Check for bearish divergence (price making higher highs, momentum making lower highs)
                elif self._is_bearish_divergence(window_signals):
                    divergence = {
                        'type': 'bearish_divergence',
                        'timeframe': timeframe.value[0],
                        'signals': [
                            {
                                'price': s.price_level,
                                'timestamp': s.timestamp.isoformat(),
                                'signal_type': s.signal_type.value
                            }
                            for s in window_signals
                        ],
                        'confidence': self._calculate_divergence_confidence(window_signals),
                        'timestamp': window_signals[-1].timestamp.isoformat()
                    }
                    divergences.append(divergence)
        
        return divergences
    
    def _is_bullish_divergence(
        self,
        signals: List[TimeframeSignal]
    ) -> bool:
        """
        Check for bullish divergence pattern
        
        Args:
            signals: List of consecutive signals
            
        Returns:
            True if bullish divergence detected
        """
        if len(signals) < 2:
            return False
        
        # Check if price is making lower lows
        prices = [s.price_level for s in signals]
        price_lows = [prices[i] for i in range(1, len(prices)-1) 
                     if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
        
        # Check if momentum is making higher lows
        momentums = [s.strength for s in signals]
        momentum_lows = [momentums[i] for i in range(1, len(momentums)-1) 
                        if momentums[i] < momentums[i-1] and momentums[i] < momentums[i+1]]
        
        # Bullish divergence: lower price lows, higher momentum lows
        if len(price_lows) >= 2 and len(momentum_lows) >= 2:
            price_decline = (price_lows[0] - price_lows[-1]) / price_lows[0]
            momentum_improvement = (momentum_lows[-1] - momentum_lows[0]) / momentum_lows[0]
            
            return (price_decline > self.divergence_threshold and 
                   momentum_improvement > self.divergence_threshold)
        
        return False
    
    def _is_bearish_divergence(
        self,
        signals: List[TimeframeSignal]
    ) -> bool:
        """
        Check for bearish divergence pattern
        
        Args:
            signals: List of consecutive signals
            
        Returns:
            True if bearish divergence detected
        """
        if len(signals) < 2:
            return False
        
        # Check if price is making higher highs
        prices = [s.price_level for s in signals]
        price_highs = [prices[i] for i in range(1, len(prices)-1) 
                      if prices[i] > prices[i-1] and prices[i] > prices[i+1]]
        
        # Check if momentum is making lower highs
        momentums = [s.strength for s in signals]
        momentum_highs = [momentums[i] for i in range(1, len(momentums)-1) 
                         if momentums[i] > momentums[i-1] and momentums[i] > momentums[i+1]]
        
        # Bearish divergence: higher price highs, lower momentum highs
        if len(price_highs) >= 2 and len(momentum_highs) >= 2:
            price_advance = (price_highs[-1] - price_highs[0]) / price_highs[0]
            momentum_decline = (momentum_highs[0] - momentum_highs[-1]) / momentum_highs[0]
            
            return (price_advance > self.divergence_threshold and 
                   momentum_decline > self.divergence_threshold)
        
        return False
    
    def _calculate_divergence_confidence(
        self,
        signals: List[TimeframeSignal]
    ) -> float:
        """
        Calculate confidence in divergence pattern
        
        Args:
            signals: List of signals forming divergence
            
        Returns:
            Confidence level (0.0 - 1.0)
        """
        if len(signals) < 2:
            return 0.0
        
        # Weight factors
        strength_weight = 0.4
        confidence_weight = 0.3
        timeframe_weight = 0.2
        alignment_weight = 0.1
        
        # Calculate components
        avg_strength = np.mean([s.strength for s in signals])
        avg_confidence = np.mean([s.confidence for s in signals])
        
        # Timeframe factor
        timeframe_factors = []
        for s in signals:
            timeframe_map = {
                Timeframe.ONE_MINUTE: 0.1,
                Timeframe.FIVE_MINUTE: 0.2,
                Timeframe.FIFTEEN_MINUTE: 0.3,
                Timeframe.THIRTY_MINUTE: 0.4,
                Timeframe.ONE_HOUR: 0.6,
                Timeframe.TWO_HOUR: 0.7,
                Timeframe.FOUR_HOUR: 0.8,
                Timeframe.SIX_HOUR: 0.85,
                Timeframe.TWELVE_HOUR: 0.9,
                Timeframe.ONE_DAY: 1.0,
                Timeframe.THREE_DAY: 1.1,
                Timeframe.ONE_WEEK: 1.2,
                Timeframe.ONE_MONTH: 1.3
            }
            timeframe_factors.append(timeframe_map.get(s.timeframe, 0.5))
        
        timeframe_factor = np.mean(timeframe_factors) if timeframe_factors else 0.5
        
        # Alignment factor (how consistent the signals are)
        bullish_count = sum(1 for s in signals if s.signal_type == SignalType.BULLISH)
        bearish_count = sum(1 for s in signals if s.signal_type == SignalType.BEARISH)
        alignment_factor = 1.0 - (abs(bullish_count - bearish_count) / len(signals))
        
        # Calculate weighted confidence
        confidence = (
            avg_strength * strength_weight +
            avg_confidence * confidence_weight +
            timeframe_factor * timeframe_weight +
            alignment_factor * alignment_weight
        )
        
        return min(1.0, confidence)
    
    def _extract_key_levels(
        self,
        confluence_zones: List[ConfluenceZone]
    ) -> List[Dict[str, Any]]:
        """
        Extract key support/resistance levels
        
        Args:
            confluence_zones: List of confluence zones
            
        Returns:
            List of key levels
        """
        key_levels = []
        
        for zone in confluence_zones:
            level = {
                'price': zone.price_level,
                'confluence_score': zone.confluence_score,
                'confluence_level': zone.confluence_level.value,
                'support_resistance': zone.support_resistance,
                'harmonic_alignment': zone.harmonic_alignment,
                'divergence_present': zone.divergence_present,
                'timeframes': [s.timeframe.value[0] for s in zone.timeframe_signals],
                'timestamp': zone.timestamp.isoformat()
            }
            key_levels.append(level)
        
        # Sort by confluence score
        key_levels.sort(key=lambda x: x['confluence_score'], reverse=True)
        
        return key_levels[:20]  # Return top 20 levels
    
    def get_trading_opportunities(self) -> List[Dict[str, Any]]:
        """
        Get trading opportunities based on multi-timeframe analysis
        
        Returns:
            List of trading opportunities
        """
        if not self.analysis_history:
            return []
        
        latest_analysis = self.analysis_history[-1]
        opportunities = []
        
        # High confluence zones
        for zone in latest_analysis.confluence_zones:
            if zone.confluence_score >= 6.0:  # Strong confluence
                opportunity = {
                    'type': 'confluence_zone',
                    'price': zone.price_level,
                    'direction': 'buy' if zone.support_resistance == 'support' else 'sell',
                    'confidence': min(1.0, zone.confluence_score / 10.0),
                    'strength': zone.confluence_score,
                    'reason': f'High confluence zone ({zone.confluence_level.value})',
                    'support_resistance': zone.support_resistance,
                    'harmonic': zone.harmonic_alignment,
                    'divergence': zone.divergence_present,
                    'timeframes': len(zone.timeframe_signals),
                    'timestamp': zone.timestamp.isoformat()
                }
                opportunities.append(opportunity)
        
        # Harmonic patterns
        for pattern in latest_analysis.harmonic_patterns:
            if pattern.get('confidence', 0) > 0.7:  # High confidence
                opportunity = {
                    'type': 'harmonic_pattern',
                    'price': pattern['points'][-1]['price'],  # Last point
                    'direction': 'buy',  # Simplified
                    'confidence': pattern['confidence'],
                    'strength': pattern['confidence'] * 10,
                    'reason': f'{pattern["type"].upper()} harmonic pattern',
                    'pattern_type': pattern['type'],
                    'timeframe': pattern['timeframe'],
                    'timestamp': pattern['timestamp']
                }
                opportunities.append(opportunity)
        
        # Divergences
        for div in latest_analysis.divergences:
            if div.get('confidence', 0) > 0.6:  # Medium-high confidence
                opportunity = {
                    'type': 'divergence',
                    'price': div['signals'][-1]['price'],  # Last signal price
                    'direction': 'buy' if div['type'] == 'bullish_divergence' else 'sell',
                    'confidence': div['confidence'],
                    'strength': div['confidence'] * 8,
                    'reason': f'{div["type"].replace("_", " ").title()}',
                    'divergence_type': div['type'],
                    'timeframe': div['timeframe'],
                    'timestamp': div['timestamp']
                }
                opportunities.append(opportunity)
        
        # Sort by confidence
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return opportunities
    
    def get_market_outlook(self) -> Dict[str, Any]:
        """
        Get overall market outlook
        
        Returns:
            Dictionary with market outlook
        """
        if not self.analysis_history:
            return {
                'trend': 'neutral',
                'trend_strength': 0.0,
                'momentum': 0.0,
                'outlook': 'neutral',
                'confidence': 0.0,
                'key_levels': [],
                'opportunities': 0
            }
        
        latest_analysis = self.analysis_history[-1]
        
        # Determine outlook
        if latest_analysis.trend_strength > 0.7:
            if latest_analysis.dominant_trend == SignalType.BULLISH:
                outlook = 'strong_bullish'
            elif latest_analysis.dominant_trend == SignalType.BEARISH:
                outlook = 'strong_bearish'
            else:
                outlook = 'neutral'
        elif latest_analysis.trend_strength > 0.4:
            if latest_analysis.dominant_trend == SignalType.BULLISH:
                outlook = 'moderate_bullish'
            elif latest_analysis.dominant_trend == SignalType.BEARISH:
                outlook = 'moderate_bearish'
            else:
                outlook = 'neutral'
        else:
            outlook = 'neutral'
        
        # Determine overall trend
        if latest_analysis.dominant_trend == SignalType.BULLISH:
            trend = 'bullish'
        elif latest_analysis.dominant_trend == SignalType.BEARISH:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Get key levels
        key_levels = [
            {
                'price': level['price'],
                'type': level['support_resistance'],
                'strength': level['confluence_score'],
                'harmonic': level['harmonic_alignment']
            }
            for level in latest_analysis.key_levels[:5]  # Top 5 levels
        ]
        
        # Count opportunities
        opportunities = self.get_trading_opportunities()
        
        return {
            'trend': trend,
            'trend_strength': float(latest_analysis.trend_strength),
            'momentum': float(latest_analysis.momentum_alignment),
            'outlook': outlook,
            'confidence': float(latest_analysis.trend_strength),
            'key_levels': key_levels,
            'opportunities': len(opportunities),
            'timestamp': latest_analysis.timestamp.isoformat()
        }

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    mta = MultiTimeframeAnalyzer(
        symbol="BTCUSDT",
        primary_timeframe=Timeframe.ONE_HOUR,
        timeframes_to_analyze=[
            Timeframe.FIFTEEN_MINUTE,
            Timeframe.ONE_HOUR,
            Timeframe.FOUR_HOUR,
            Timeframe.ONE_DAY,
            Timeframe.ONE_WEEK
        ]
    )
    
    print("=== MULTI-TIMEFRAME ANALYZER EXAMPLE ===\n")
    
    # Generate sample data for different timeframes
    np.random.seed(42)
    base_price = 40000.0
    
    # Create sample data for each timeframe
    timeframes_data = {
        Timeframe.FIFTEEN_MINUTE: 240,   # 15m * 240 = 60 hours
        Timeframe.ONE_HOUR: 168,         # 1h * 168 = 1 week
        Timeframe.FOUR_HOUR: 84,         # 4h * 84 = 2 weeks
        Timeframe.ONE_DAY: 90,           # 1d * 90 = 3 months
        Timeframe.ONE_WEEK: 52           # 1w * 52 = 1 year
    }
    
    for timeframe, periods in timeframes_data.items():
        # Generate realistic price data
        returns = np.random.normal(0.0001, 0.02, periods)  # Small positive drift
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        opens = prices[:-1]
        closes = prices[1:]
        highs = np.maximum(opens, closes) * np.random.uniform(1.001, 1.005, len(opens))
        lows = np.minimum(opens, closes) * np.random.uniform(0.995, 0.999, len(opens))
        volumes = np.random.exponential(1000, len(opens))  # Volume data
        
        # Create DataFrame
        timestamps = pd.date_range(
            end=pd.Timestamp.now(),
            periods=len(opens),
            freq=f"{timeframe.value[1]}T"  # Use timeframe minutes
        )
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=timestamps)
        
        # Add to analyzer
        mta.add_timeframe_data(timeframe, df)
    
    print("Data loaded for timeframes:")
    for tf in mta.timeframe_data.keys():
        print(f"  - {tf.value[0]}: {len(mta.timeframe_data[tf])} candles")
    
    # Perform analysis
    print("\nPerforming multi-timeframe analysis...")
    analysis = mta.analyze_all_timeframes()
    
    print(f"\nAnalysis Results:")
    print(f"  Symbol: {analysis.symbol}")
    print(f"  Primary Timeframe: {analysis.primary_timeframe.value[0]}")
    print(f"  Dominant Trend: {analysis.dominant_trend.value}")
    print(f"  Trend Strength: {analysis.trend_strength:.2f}")
    print(f"  Momentum Alignment: {analysis.momentum_alignment:.2f}")
    print(f"  Confluence Zones: {len(analysis.confluence_zones)}")
    print(f"  Harmonic Patterns: {len(analysis.harmonic_patterns)}")
    print(f"  Divergences: {len(analysis.divergences)}")
    print(f"  Key Levels: {len(analysis.key_levels)}")
    
    # Show top confluence zones
    print(f"\nTop Confluence Zones:")
    for i, zone in enumerate(analysis.confluence_zones[:3]):
        print(f"  {i+1}. ${zone.price_level:.2f} - Score: {zone.confluence_score:.1f} ({zone.confluence_level.value})")
        print(f"     Type: {zone.support_resistance}, Harmonic: {zone.harmonic_alignment}, Divergence: {zone.divergence_present}")
        print(f"     Timeframes: {[s.timeframe.value[0] for s in zone.timeframe_signals[:3]]}")
    
    # Get trading opportunities
    opportunities = mta.get_trading_opportunities()
    print(f"\nTrading Opportunities ({len(opportunities)}):")
    for i, opp in enumerate(opportunities[:3]):
        print(f"  {i+1}. {opp['type'].title()} at ${opp['price']:.2f}")
        print(f"     Direction: {opp['direction'].upper()}, Confidence: {opp['confidence']:.2f}")
        print(f"     Reason: {opp['reason']}")
    
    # Get market outlook
    outlook = mta.get_market_outlook()
    print(f"\nMarket Outlook:")
    print(f"  Trend: {outlook['trend'].title()}")
    print(f"  Strength: {outlook['trend_strength']:.2f}")
    print(f"  Momentum: {outlook['momentum']:.2f}")
    print(f"  Outlook: {outlook['outlook'].replace('_', ' ').title()}")
    print(f"  Confidence: {outlook['confidence']:.2f}")
    print(f"  Opportunities: {outlook['opportunities']}")
    
    print(f"\nKey Support/Resistance Levels:")
    for i, level in enumerate(outlook['key_levels'][:3]):
        print(f"  {i+1}. ${level['price']:.2f} - {level['type'].title()} (Strength: {level['strength']:.1f})")