"""
ORDER FLOW ANALYZER
===================

Advanced order flow analysis for crypto trading including:
- Volume Profile
- Delta Flow Analysis
- Footprint Charts
- Cumulative Volume Delta
- Market Microstructure Analysis

Features:
- Real-time order flow processing
- Volume clustering and identification
- Institutional order detection
- Market impact assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import defaultdict, deque
import math

class OrderFlowEvent(Enum):
    """Order flow event types"""
    VOLUME_SPIKE = "volume_spike"
    DELTA_IMBALANCE = "delta_imbalance"
    CLUSTER_DETECTION = "cluster_detection"
    INSTITUTIONAL_ORDER = "institutional_order"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    STOP_RUN = "stop_run"
    FAKEY = "fakey"
    CHOCH = "choch"  # Change of Character

class MarketRegime(Enum):
    """Market regime based on order flow"""
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    INSTITUTIONAL = "institutional"
    RETAIL = "retail"

@dataclass
class VolumeProfile:
    """Volume profile at price level"""
    price: float
    volume: float
    buy_volume: float
    sell_volume: float
    delta: float  # Buy - Sell
    poc: bool = False  # Point of Control
    value_area: bool = False  # Within value area

@dataclass
class OrderFlowEventDetail:
    """Detailed order flow event"""
    event_type: OrderFlowEvent
    price: float
    volume: float
    delta: float
    timestamp: pd.Timestamp
    confidence: float  # 0.0 - 1.0
    strength: float   # Event strength
    metadata: Dict[str, Any]

@dataclass
class MarketMicrostructure:
    """Market microstructure metrics"""
    bid_ask_spread: float
    market_depth: float
    order_book_imbalance: float
    volume_imbalance: float
    delta_imbalance: float
    liquidity_levels: List[Tuple[float, float]]  # price, liquidity
    regime: MarketRegime

class OrderFlowAnalyzer:
    """
    Advanced Order Flow Analyzer
    
    Analyzes market microstructure and order flow patterns to:
    1. Identify institutional activity
    2. Detect volume clusters and liquidity zones
    3. Assess market impact and direction
    4. Recognize order flow patterns
    5. Generate actionable insights
    """
    
    def __init__(
        self,
        value_area_percentage: float = 0.70,
        volume_spike_threshold: float = 2.0,  # 2x average volume
        delta_imbalance_threshold: float = 0.6,  # 60% imbalance
        cluster_detection_window: int = 20,
        institutional_order_threshold: float = 1000000,  # $1M minimum
        lookback_periods: int = 100
    ):
        self.value_area_percentage = value_area_percentage
        self.volume_spike_threshold = volume_spike_threshold
        self.delta_imbalance_threshold = delta_imbalance_threshold
        self.cluster_detection_window = cluster_detection_window
        self.institutional_order_threshold = institutional_order_threshold
        self.lookback_periods = lookback_periods
        
        # Storage for analysis
        self.volume_profiles: List[VolumeProfile] = []
        self.order_flow_events: List[OrderFlowEventDetail] = []
        self.microstructure_history: List[MarketMicrostructure] = []
        self.price_history: deque = deque(maxlen=lookback_periods)
        self.volume_history: deque = deque(maxlen=lookback_periods)
        self.delta_history: deque = deque(maxlen=lookback_periods)
        
        # Real-time tracking
        self.current_profile: Dict[float, VolumeProfile] = {}
        self.session_high = float('-inf')
        self.session_low = float('inf')
        self.point_of_control = 0.0
        self.value_area_high = 0.0
        self.value_area_low = 0.0
        
    def update_with_tick_data(
        self,
        price: float,
        volume: float,
        side: str,  # 'buy' or 'sell'
        timestamp: pd.Timestamp
    ):
        """
        Update analyzer with tick data
        
        Args:
            price: Transaction price
            volume: Transaction volume
            side: 'buy' or 'sell'
            timestamp: Transaction timestamp
        """
        # Update history
        self.price_history.append(price)
        self.volume_history.append(volume)
        
        # Calculate delta
        delta = volume if side.lower() == 'buy' else -volume
        self.delta_history.append(delta)
        
        # Update session bounds
        self.session_high = max(self.session_high, price)
        self.session_low = min(self.session_low, price)
        
        # Update volume profile
        price_level = self._round_to_price_level(price)
        
        if price_level not in self.current_profile:
            self.current_profile[price_level] = VolumeProfile(
                price=price_level,
                volume=0.0,
                buy_volume=0.0,
                sell_volume=0.0,
                delta=0.0
            )
        
        profile = self.current_profile[price_level]
        profile.volume += volume
        if side.lower() == 'buy':
            profile.buy_volume += volume
        else:
            profile.sell_volume += volume
        profile.delta += delta
        
        # Update POC if needed
        if profile.volume > self.current_profile.get(self.point_of_control, VolumeProfile(0, 0, 0, 0, 0)).volume:
            self.point_of_control = price_level
        
        # Detect events
        self._detect_order_flow_events(price, volume, delta, side, timestamp)
    
    def _round_to_price_level(self, price: float, tick_size: float = 50.0) -> float:
        """
        Round price to nearest price level
        
        Args:
            price: Raw price
            tick_size: Price level increment
            
        Returns:
            Rounded price level
        """
        return round(price / tick_size) * tick_size
    
    def _detect_order_flow_events(
        self,
        price: float,
        volume: float,
        delta: float,
        side: str,
        timestamp: pd.Timestamp
    ):
        """
        Detect order flow events from tick data
        
        Args:
            price: Transaction price
            volume: Transaction volume
            delta: Volume delta (buy - sell)
            side: Transaction side
            timestamp: Transaction timestamp
        """
        # Volume spike detection
        if len(self.volume_history) >= 10:
            avg_volume = np.mean(list(self.volume_history)[-10:])
            if volume > avg_volume * self.volume_spike_threshold:
                event = OrderFlowEventDetail(
                    event_type=OrderFlowEvent.VOLUME_SPIKE,
                    price=price,
                    volume=volume,
                    delta=delta,
                    timestamp=timestamp,
                    confidence=min(1.0, volume / (avg_volume * self.volume_spike_threshold)),
                    strength=volume / avg_volume,
                    metadata={
                        'avg_volume': avg_volume,
                        'spike_ratio': volume / avg_volume
                    }
                )
                self.order_flow_events.append(event)
        
        # Delta imbalance detection
        if len(self.delta_history) >= 5:
            recent_deltas = list(self.delta_history)[-5:]
            total_volume = sum(abs(d) for d in recent_deltas)
            net_delta = sum(recent_deltas)
            
            if total_volume > 0:
                delta_ratio = abs(net_delta) / total_volume
                if delta_ratio > self.delta_imbalance_threshold:
                    event = OrderFlowEventDetail(
                        event_type=OrderFlowEvent.DELTA_IMBALANCE,
                        price=price,
                        volume=volume,
                        delta=delta,
                        timestamp=timestamp,
                        confidence=min(1.0, delta_ratio / self.delta_imbalance_threshold),
                        strength=delta_ratio,
                        metadata={
                            'net_delta': net_delta,
                            'total_volume': total_volume,
                            'delta_ratio': delta_ratio,
                            'dominant_side': 'buy' if net_delta > 0 else 'sell'
                        }
                    )
                    self.order_flow_events.append(event)
        
        # Institutional order detection
        if volume * price > self.institutional_order_threshold:
            event = OrderFlowEventDetail(
                event_type=OrderFlowEvent.INSTITUTIONAL_ORDER,
                price=price,
                volume=volume,
                delta=delta,
                timestamp=timestamp,
                confidence=1.0,
                strength=volume * price / self.institutional_order_threshold,
                metadata={
                    'order_value': volume * price,
                    'side': side,
                    'threshold': self.institutional_order_threshold
                }
            )
            self.order_flow_events.append(event)
    
    def calculate_volume_profile(
        self,
        price_levels: Optional[List[float]] = None
    ) -> List[VolumeProfile]:
        """
        Calculate volume profile from current data
        
        Args:
            price_levels: Optional predefined price levels
            
        Returns:
            List of volume profiles
        """
        if not self.current_profile:
            return []
        
        # Convert to list and sort
        profiles = list(self.current_profile.values())
        profiles.sort(key=lambda x: x.price)
        
        # Calculate value area
        total_volume = sum(p.volume for p in profiles)
        target_volume = total_volume * self.value_area_percentage
        
        # Find POC and value area
        poc_index = 0
        max_volume = 0
        for i, profile in enumerate(profiles):
            if profile.volume > max_volume:
                max_volume = profile.volume
                poc_index = i
                self.point_of_control = profile.price
        
        # Calculate value area by expanding from POC
        value_area_volume = profiles[poc_index].volume
        upper_index = poc_index
        lower_index = poc_index
        
        while value_area_volume < target_volume and (upper_index < len(profiles) - 1 or lower_index > 0):
            # Expand both directions
            next_upper_volume = profiles[upper_index + 1].volume if upper_index < len(profiles) - 1 else 0
            next_lower_volume = profiles[lower_index - 1].volume if lower_index > 0 else 0
            
            if next_upper_volume >= next_lower_volume and upper_index < len(profiles) - 1:
                upper_index += 1
                value_area_volume += profiles[upper_index].volume
            elif lower_index > 0:
                lower_index -= 1
                value_area_volume += profiles[lower_index].volume
            else:
                break
        
        # Mark value area profiles
        for i in range(lower_index, upper_index + 1):
            profiles[i].value_area = True
        
        self.value_area_high = profiles[upper_index].price
        self.value_area_low = profiles[lower_index].price
        
        return profiles
    
    def calculate_cumulative_delta(self, window: int = 50) -> float:
        """
        Calculate cumulative volume delta
        
        Args:
            window: Lookback window
            
        Returns:
            Cumulative delta
        """
        if len(self.delta_history) < window:
            return sum(self.delta_history)
        
        return sum(list(self.delta_history)[-window:])
    
    def analyze_market_microstructure(self) -> MarketMicrostructure:
        """
        Analyze current market microstructure
        
        Returns:
            MarketMicrostructure object with metrics
        """
        if len(self.price_history) < 2 or len(self.volume_history) < 2:
            return MarketMicrostructure(
                bid_ask_spread=0.0,
                market_depth=0.0,
                order_book_imbalance=0.0,
                volume_imbalance=0.0,
                delta_imbalance=0.0,
                liquidity_levels=[],
                regime=MarketRegime.NORMAL
            )
        
        # Calculate metrics
        prices = list(self.price_history)
        volumes = list(self.volume_history)
        deltas = list(self.delta_history)
        
        # Bid-Ask spread estimation
        spreads = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        avg_spread = np.mean(spreads) if spreads else 0.0
        
        # Market depth (volume-weighted)
        if volumes:
            market_depth = np.mean(volumes) * len(volumes)
        else:
            market_depth = 0.0
        
        # Order book imbalance
        if len(deltas) >= 10:
            recent_deltas = deltas[-10:]
            total_volume = sum(abs(d) for d in recent_deltas)
            net_delta = sum(recent_deltas)
            ob_imbalance = net_delta / total_volume if total_volume > 0 else 0.0
        else:
            ob_imbalance = 0.0
        
        # Volume imbalance
        if len(volumes) >= 10:
            recent_volumes = volumes[-10:]
            avg_volume = np.mean(recent_volumes)
            current_volume = recent_volumes[-1]
            volume_imbalance = (current_volume - avg_volume) / avg_volume if avg_volume > 0 else 0.0
        else:
            volume_imbalance = 0.0
        
        # Delta imbalance
        delta_imbalance = ob_imbalance
        
        # Liquidity levels (simplified)
        liquidity_levels = []
        if self.current_profile:
            # Top 5 volume levels
            sorted_profiles = sorted(
                self.current_profile.values(),
                key=lambda x: x.volume,
                reverse=True
            )[:5]
            
            for profile in sorted_profiles:
                liquidity_levels.append((profile.price, profile.volume))
        
        # Market regime determination
        regime = self._determine_market_regime(
            avg_spread, volume_imbalance, delta_imbalance
        )
        
        microstructure = MarketMicrostructure(
            bid_ask_spread=avg_spread,
            market_depth=market_depth,
            order_book_imbalance=ob_imbalance,
            volume_imbalance=volume_imbalance,
            delta_imbalance=delta_imbalance,
            liquidity_levels=liquidity_levels,
            regime=regime
        )
        
        self.microstructure_history.append(microstructure)
        return microstructure
    
    def _determine_market_regime(
        self,
        bid_ask_spread: float,
        volume_imbalance: float,
        delta_imbalance: float
    ) -> MarketRegime:
        """
        Determine current market regime
        
        Args:
            bid_ask_spread: Current bid-ask spread
            volume_imbalance: Volume imbalance metric
            delta_imbalance: Delta imbalance metric
            
        Returns:
            MarketRegime enum
        """
        # High volatility regime
        if bid_ask_spread > 100:  # Adjust threshold based on asset
            return MarketRegime.HIGH_VOLATILITY
        
        # Low liquidity regime
        if abs(volume_imbalance) > 2.0:  # 200% above average volume
            return MarketRegime.LOW_LIQUIDITY
        
        # Institutional regime (large delta imbalances)
        if abs(delta_imbalance) > 0.7:  # 70% buy/sell dominance
            return MarketRegime.INSTITUTIONAL
        
        # Retail regime (small trades, balanced flow)
        if abs(delta_imbalance) < 0.3 and abs(volume_imbalance) < 0.5:
            return MarketRegime.RETAIL
        
        return MarketRegime.NORMAL
    
    def detect_clusters_and_liquidity(
        self,
        window: int = 20
    ) -> List[OrderFlowEventDetail]:
        """
        Detect volume clusters and liquidity zones
        
        Args:
            window: Analysis window
            
        Returns:
            List of cluster/liquidity events
        """
        clusters = []
        
        if not self.current_profile:
            return clusters
        
        # Convert to sorted list
        profiles = sorted(
            self.current_profile.values(),
            key=lambda x: x.price
        )
        
        # Find significant volume levels
        total_volume = sum(p.volume for p in profiles)
        avg_volume = total_volume / len(profiles) if profiles else 0
        
        for profile in profiles:
            # Significant volume level
            if profile.volume > avg_volume * 1.5:
                cluster_event = OrderFlowEventDetail(
                    event_type=OrderFlowEvent.CLUSTER_DETECTION,
                    price=profile.price,
                    volume=profile.volume,
                    delta=profile.delta,
                    timestamp=pd.Timestamp.now(),
                    confidence=min(1.0, profile.volume / (avg_volume * 2)),
                    strength=profile.volume / avg_volume,
                    metadata={
                        'type': 'volume_cluster',
                        'relative_volume': profile.volume / avg_volume
                    }
                )
                clusters.append(cluster_event)
            
            # High delta level (potential liquidity)
            abs_delta = abs(profile.delta)
            avg_delta = np.mean([abs(p.delta) for p in profiles]) if profiles else 0
            if abs_delta > avg_delta * 2:
                liquidity_event = OrderFlowEventDetail(
                    event_type=OrderFlowEvent.CLUSTER_DETECTION,
                    price=profile.price,
                    volume=profile.volume,
                    delta=profile.delta,
                    timestamp=pd.Timestamp.now(),
                    confidence=min(1.0, abs_delta / (avg_delta * 3)),
                    strength=abs_delta / avg_delta,
                    metadata={
                        'type': 'liquidity_zone',
                        'delta_strength': abs_delta / avg_delta
                    }
                )
                clusters.append(liquidity_event)
        
        return clusters
    
    def generate_order_flow_insights(self) -> Dict[str, Any]:
        """
        Generate comprehensive order flow insights
        
        Returns:
            Dictionary with insights and recommendations
        """
        insights = {
            'timestamp': pd.Timestamp.now(),
            'market_regime': None,
            'volume_profile': {},
            'key_levels': [],
            'events': [],
            'recommendations': [],
            'risk_metrics': {}
        }
        
        # Get current microstructure
        microstructure = self.analyze_market_microstructure()
        insights['market_regime'] = microstructure.regime.value
        
        # Volume profile insights
        profiles = self.calculate_volume_profile()
        if profiles:
            insights['volume_profile'] = {
                'poc': self.point_of_control,
                'value_area_high': self.value_area_high,
                'value_area_low': self.value_area_low,
                'session_high': self.session_high,
                'session_low': self.session_low
            }
            
            # Key support/resistance levels
            poc_profile = min(profiles, key=lambda x: abs(x.price - self.point_of_control))
            va_profiles = [p for p in profiles if p.value_area]
            
            if va_profiles:
                insights['key_levels'] = [
                    {
                        'type': 'poc',
                        'price': self.point_of_control,
                        'volume': poc_profile.volume,
                        'strength': poc_profile.volume / max(p.volume for p in profiles)
                    },
                    {
                        'type': 'value_area_high',
                        'price': self.value_area_high,
                        'volume': max(p.volume for p in va_profiles if p.price == self.value_area_high),
                        'strength': 1.0
                    },
                    {
                        'type': 'value_area_low',
                        'price': self.value_area_low,
                        'volume': max(p.volume for p in va_profiles if p.price == self.value_area_low),
                        'strength': 1.0
                    }
                ]
        
        # Recent events
        if self.order_flow_events:
            recent_events = self.order_flow_events[-10:]  # Last 10 events
            insights['events'] = [
                {
                    'type': event.event_type.value,
                    'price': event.price,
                    'volume': event.volume,
                    'delta': event.delta,
                    'confidence': event.confidence,
                    'strength': event.strength,
                    'timestamp': event.timestamp.isoformat()
                }
                for event in recent_events
            ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(microstructure)
        insights['recommendations'] = recommendations
        
        # Risk metrics
        insights['risk_metrics'] = {
            'bid_ask_spread': microstructure.bid_ask_spread,
            'market_depth': microstructure.market_depth,
            'cumulative_delta': self.calculate_cumulative_delta(),
            'volatility_regime': self._assess_volatility_regime()
        }
        
        return insights
    
    def _generate_recommendations(
        self,
        microstructure: MarketMicrostructure
    ) -> List[Dict[str, Any]]:
        """
        Generate trading recommendations based on order flow
        
        Args:
            microstructure: Current market microstructure
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Regime-based recommendations
        if microstructure.regime == MarketRegime.HIGH_VOLATILITY:
            recommendations.append({
                'type': 'caution',
                'message': 'High volatility regime - reduce position size',
                'priority': 'high',
                'confidence': 0.8
            })
        elif microstructure.regime == MarketRegime.LOW_LIQUIDITY:
            recommendations.append({
                'type': 'caution',
                'message': 'Low liquidity regime - wider stops recommended',
                'priority': 'high',
                'confidence': 0.7
            })
        elif microstructure.regime == MarketRegime.INSTITUTIONAL:
            recommendations.append({
                'type': 'opportunity',
                'message': 'Institutional activity detected - follow strong moves',
                'priority': 'high',
                'confidence': 0.9
            })
        
        # Delta-based recommendations
        cum_delta = self.calculate_cumulative_delta()
        if abs(cum_delta) > 1000000:  # $1M threshold
            direction = 'bullish' if cum_delta > 0 else 'bearish'
            strength = min(1.0, abs(cum_delta) / 5000000)  # Normalize to $5M
            
            recommendations.append({
                'type': 'momentum',
                'message': f'Strong {direction} momentum detected',
                'priority': 'medium',
                'confidence': strength,
                'metadata': {
                    'cumulative_delta': cum_delta,
                    'direction': direction
                }
            })
        
        # Volume-based recommendations
        if len(self.volume_history) >= 5:
            recent_volume = np.mean(list(self.volume_history)[-5:])
            avg_volume = np.mean(list(self.volume_history))
            
            if recent_volume > avg_volume * 1.5:
                recommendations.append({
                    'type': 'breakout',
                    'message': 'Above average volume - potential breakout',
                    'priority': 'medium',
                    'confidence': min(1.0, recent_volume / (avg_volume * 2)),
                    'metadata': {
                        'recent_volume': recent_volume,
                        'average_volume': avg_volume,
                        'ratio': recent_volume / avg_volume
                    }
                })
        
        return recommendations
    
    def _assess_volatility_regime(self) -> str:
        """
        Assess current volatility regime
        
        Returns:
            Volatility regime string
        """
        if len(self.price_history) < 10:
            return 'normal'
        
        prices = list(self.price_history)[-10:]
        returns = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 'normal'
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # High volatility if returns are large
        if avg_return > 0.02:  # 2% average move
            return 'high'
        elif avg_return < 0.005:  # 0.5% average move
            return 'low'
        else:
            return 'normal'
    
    def reset_session(self):
        """
        Reset session data for new trading session
        """
        self.current_profile.clear()
        self.session_high = float('-inf')
        self.session_low = float('inf')
        self.point_of_control = 0.0
        self.value_area_high = 0.0
        self.value_area_low = 0.0
        self.order_flow_events.clear()
        self.microstructure_history.clear()
        self.price_history.clear()
        self.volume_history.clear()
        self.delta_history.clear()

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    ofa = OrderFlowAnalyzer()
    
    print("=== ORDER FLOW ANALYZER EXAMPLE ===\n")
    
    # Simulate tick data
    np.random.seed(42)
    base_price = 40000.0
    timestamps = pd.date_range('2023-01-01', periods=1000, freq='S')  # 1000 seconds
    
    # Generate realistic tick data
    for i, timestamp in enumerate(timestamps):
        # Random price movement
        price_change = np.random.normal(0, 50)  # $50 std dev
        current_price = base_price + price_change + (i * 0.1)  # Slight uptrend
        
        # Volume with occasional spikes
        volume = np.random.exponential(0.5)  # Most trades small
        if np.random.random() < 0.05:  # 5% chance of large volume
            volume *= 10  # Volume spike
        
        # Random buy/sell
        side = 'buy' if np.random.random() < 0.52 else 'sell'  # Slight buy bias
        
        # Update analyzer
        ofa.update_with_tick_data(
            price=current_price,
            volume=volume,
            side=side,
            timestamp=timestamp
        )
    
    # Generate insights
    insights = ofa.generate_order_flow_insights()
    
    print("Market Insights:")
    print(f"  Regime: {insights['market_regime']}")
    print(f"  POC: ${insights['volume_profile'].get('poc', 0):.2f}")
    print(f"  VA High: ${insights['volume_profile'].get('value_area_high', 0):.2f}")
    print(f"  VA Low: ${insights['volume_profile'].get('value_area_low', 0):.2f}")
    
    print(f"\nKey Levels:")
    for level in insights['key_levels']:
        print(f"  {level['type']}: ${level['price']:.2f} (strength: {level['strength']:.2f})")
    
    print(f"\nRecent Events ({len(insights['events'])}):")
    for event in insights['events'][:3]:  # Show first 3
        print(f"  {event['type']}: ${event['price']:.2f} (confidence: {event['confidence']:.2f})")
    
    print(f"\nRecommendations ({len(insights['recommendations'])}):")
    for rec in insights['recommendations']:
        print(f"  [{rec['priority'].upper()}] {rec['message']} (confidence: {rec['confidence']:.2f})")
    
    print(f"\nRisk Metrics:")
    for metric, value in insights['risk_metrics'].items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")