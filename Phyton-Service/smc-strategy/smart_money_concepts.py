"""
SMART MONEY CONCEPTS (SMC) TRADING STRATEGY
==========================================

Advanced implementation of Smart Money Concepts including:
- Liquidity Zones
- Order Blocks
- Structural Breaks
- Mitigation Blocks
- Premium/Discount Zones

Based on ICT (Inner Circle Trader) methodology with mathematical precision
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

class SMCLevelType(Enum):
    """SMC level types"""
    LIQUIDITY_HIGH = "liquidity_high"
    LIQUIDITY_LOW = "liquidity_low"
    ORDER_BLOCK_BULLISH = "order_block_bullish"
    ORDER_BLOCK_BEARISH = "order_block_bearish"
    STRUCTURAL_HIGH = "structural_high"
    STRUCTURAL_LOW = "structural_low"
    MITIGATION_BLOCK = "mitigation_block"
    PREMIUM_ZONE = "premium_zone"
    DISCOUNT_ZONE = "discount_zone"

@dataclass
class SMCLevel:
    """SMC level with metadata"""
    level_type: SMCLevelType
    price: float
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    timestamp: pd.Timestamp
    timeframe: str
    touched_count: int = 0
    broken_count: int = 0
    last_touched: Optional[pd.Timestamp] = None

@dataclass
class SMCTradeSetup:
    """SMC trade setup"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_zone: Tuple[float, float]  # (lower, upper)
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float  # 0.0 - 1.0
    setup_score: float  # 0.0 - 10.0
    levels_used: List[SMCLevel]
    timestamp: pd.Timestamp

class SmartMoneyConcepts:
    """
    Advanced Smart Money Concepts Trading Strategy
    
    Implements core SMC concepts with mathematical precision:
    1. Liquidity Zones - Areas where smart money takes liquidity
    2. Order Blocks - Institutional order placement zones
    3. Structural Breaks - Significant market structure changes
    4. Mitigation Blocks - Areas that mitigate structural breaks
    5. Premium/Discount Zones - Fair value deviations
    """
    
    def __init__(
        self,
        lookback_periods: int = 50,
        liquidity_sensitivity: float = 0.8,
        order_block_validity: int = 20,  # candles
        structural_significance: float = 0.02,  # 2% price move minimum
        mitigation_threshold: float = 0.5  # 50% retracement
    ):
        self.lookback_periods = lookback_periods
        self.liquidity_sensitivity = liquidity_sensitivity
        self.order_block_validity = order_block_validity
        self.structural_significance = structural_significance
        self.mitigation_threshold = mitigation_threshold
        
        # Stored levels
        self.levels: List[SMCLevel] = []
        self.historical_levels: List[SMCLevel] = []
        
    def identify_liquidity_zones(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Tuple[List[SMCLevel], List[SMCLevel]]:
        """
        Identify liquidity high and low zones
        
        Liquidity zones are areas where:
        - High volume concentration occurs
        - Sharp price movements happen
        - Market makers take liquidity
        
        Args:
            df: DataFrame with OHLCV data
            window: Lookback window for liquidity calculation
            
        Returns:
            Tuple of (liquidity_highs, liquidity_lows)
        """
        highs = []
        lows = []
        
        # Calculate volume-weighted metrics
        volume_ma = df['volume'].rolling(window=window).mean()
        price_range = df['high'] - df['low']
        
        # Liquidity measure: volume * price movement / average
        liquidity_measure = (df['volume'] * price_range) / (volume_ma * df['close'])
        normalized_liquidity = (liquidity_measure - liquidity_measure.rolling(window).min()) / \
                              (liquidity_measure.rolling(window).max() - liquidity_measure.rolling(window).min())
        
        # Identify significant liquidity events
        liquidity_threshold = self.liquidity_sensitivity
        
        for i in range(window, len(df)):
            current_liquidity = normalized_liquidity.iloc[i]
            
            if current_liquidity > liquidity_threshold:
                # Check if this is a high or low
                prev_close = df['close'].iloc[i-1]
                current_close = df['close'].iloc[i]
                
                if current_close > prev_close:  # Bullish move
                    level = SMCLevel(
                        level_type=SMCLevelType.LIQUIDITY_HIGH,
                        price=df['high'].iloc[i],
                        strength=current_liquidity,
                        confidence=min(1.0, current_liquidity),
                        timestamp=df.index[i],
                        timeframe=f"{window}c",
                        touched_count=1
                    )
                    highs.append(level)
                    
                else:  # Bearish move
                    level = SMCLevel(
                        level_type=SMCLevelType.LIQUIDITY_LOW,
                        price=df['low'].iloc[i],
                        strength=current_liquidity,
                        confidence=min(1.0, current_liquidity),
                        timestamp=df.index[i],
                        timeframe=f"{window}c",
                        touched_count=1
                    )
                    lows.append(level)
        
        return highs, lows
    
    def identify_order_blocks(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[SMCLevel], List[SMCLevel]]:
        """
        Identify bullish and bearish order blocks
        
        Order blocks are institutional order placement zones where:
        - Strong momentum moves occur
        - Little to no opposite reaction
        - Clear directional bias
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (bullish_order_blocks, bearish_order_blocks)
        """
        bullish_blocks = []
        bearish_blocks = []
        
        for i in range(self.order_block_validity, len(df)):
            # Look for strong moves
            lookback_start = max(0, i - self.order_block_validity)
            
            # Calculate move strength
            price_move = (df['close'].iloc[i] - df['close'].iloc[lookback_start]) / df['close'].iloc[lookback_start]
            
            # Check for significant move
            if abs(price_move) > self.structural_significance:
                if price_move > 0:  # Bullish move
                    # Check for minimal retracement (strong trend)
                    retracement = (df['low'].iloc[i] - df['low'].iloc[lookback_start]) / \
                                  (df['close'].iloc[i] - df['close'].iloc[lookback_start])
                    
                    if retracement < 0.3:  # Less than 30% retracement
                        # Bullish order block at the beginning of move
                        ob_price = df['low'].iloc[lookback_start:i].min()
                        
                        level = SMCLevel(
                            level_type=SMCLevelType.ORDER_BLOCK_BULLISH,
                            price=ob_price,
                            strength=abs(price_move),
                            confidence=min(1.0, abs(price_move) / 0.1),  # Normalize to 10% move
                            timestamp=df.index[lookback_start],
                            timeframe=f"{self.order_block_validity}c",
                            touched_count=1
                        )
                        bullish_blocks.append(level)
                        
                else:  # Bearish move
                    # Check for minimal retracement (strong trend)
                    retracement = (df['high'].iloc[lookback_start] - df['high'].iloc[i]) / \
                                  (df['close'].iloc[lookback_start] - df['close'].iloc[i])
                    
                    if retracement < 0.3:  # Less than 30% retracement
                        # Bearish order block at the beginning of move
                        ob_price = df['high'].iloc[lookback_start:i].max()
                        
                        level = SMCLevel(
                            level_type=SMCLevelType.ORDER_BLOCK_BEARISH,
                            price=ob_price,
                            strength=abs(price_move),
                            confidence=min(1.0, abs(price_move) / 0.1),  # Normalize to 10% move
                            timestamp=df.index[lookback_start],
                            timeframe=f"{self.order_block_validity}c",
                            touched_count=1
                        )
                        bearish_blocks.append(level)
        
        return bullish_blocks, bearish_blocks
    
    def identify_structural_levels(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[SMCLevel], List[SMCLevel]]:
        """
        Identify structural high and low levels
        
        Structural levels are significant swing points where:
        - Clear market structure changes
        - Significant price movements occur
        - Multiple touches or breaks happen
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (structural_highs, structural_lows)
        """
        structural_highs = []
        structural_lows = []
        
        # Find swing points using rolling windows
        window = 10
        
        for i in range(window, len(df) - window):
            # Check if current point is local high/low
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Local high check
            is_high = True
            for j in range(i - window, i + window + 1):
                if j != i and j < len(df) and df['high'].iloc[j] > current_high:
                    is_high = False
                    break
            
            # Local low check
            is_low = True
            for j in range(i - window, i + window + 1):
                if j != i and j < len(df) and df['low'].iloc[j] < current_low:
                    is_low = False
                    break
            
            if is_high:
                # Check significance
                prev_window = df['high'].iloc[max(0, i-50):i]
                if len(prev_window) > 0:
                    avg_high = prev_window.mean()
                    significance = (current_high - avg_high) / avg_high
                    
                    if significance > self.structural_significance:
                        level = SMCLevel(
                            level_type=SMCLevelType.STRUCTURAL_HIGH,
                            price=current_high,
                            strength=significance,
                            confidence=min(1.0, significance / 0.05),  # Normalize to 5% move
                            timestamp=df.index[i],
                            timeframe="50c",
                            touched_count=1
                        )
                        structural_highs.append(level)
            
            if is_low:
                # Check significance
                prev_window = df['low'].iloc[max(0, i-50):i]
                if len(prev_window) > 0:
                    avg_low = prev_window.mean()
                    significance = (avg_low - current_low) / avg_low
                    
                    if significance > self.structural_significance:
                        level = SMCLevel(
                            level_type=SMCLevelType.STRUCTURAL_LOW,
                            price=current_low,
                            strength=significance,
                            confidence=min(1.0, significance / 0.05),  # Normalize to 5% move
                            timestamp=df.index[i],
                            timeframe="50c",
                            touched_count=1
                        )
                        structural_lows.append(level)
        
        return structural_highs, structural_lows
    
    def identify_mitigation_blocks(
        self,
        df: pd.DataFrame,
        structural_levels: List[SMCLevel]
    ) -> List[SMCLevel]:
        """
        Identify mitigation blocks for structural levels
        
        Mitigation blocks form when price retraces a significant portion
        of a structural move, indicating potential continuation.
        
        Args:
            df: DataFrame with OHLCV data
            structural_levels: Previously identified structural levels
            
        Returns:
            List of mitigation blocks
        """
        mitigation_blocks = []
        
        for level in structural_levels:
            # Find price action after the structural level
            level_idx = df.index.get_loc(level.timestamp)
            if level_idx == -1:
                continue
                
            # Look forward for retracement
            forward_window = min(30, len(df) - level_idx - 1)
            
            if forward_window > 5:
                if level.level_type == SMCLevelType.STRUCTURAL_HIGH:
                    # Look for retracement upwards
                    max_retracement = df['low'].iloc[level_idx:level_idx+forward_window].min()
                    retracement_ratio = (level.price - max_retracement) / (level.price - df['low'].iloc[level_idx-20:level_idx].min())
                    
                    if retracement_ratio > self.mitigation_threshold:
                        # Mitigation block formed
                        mb_price = (level.price + max_retracement) / 2
                        
                        level_obj = SMCLevel(
                            level_type=SMCLevelType.MITIGATION_BLOCK,
                            price=mb_price,
                            strength=retracement_ratio,
                            confidence=min(1.0, retracement_ratio),
                            timestamp=df.index[level_idx + np.argmin(df['low'].iloc[level_idx:level_idx+forward_window])],
                            timeframe="30c",
                            touched_count=1
                        )
                        mitigation_blocks.append(level_obj)
                        
                elif level.level_type == SMCLevelType.STRUCTURAL_LOW:
                    # Look for retracement downwards
                    max_retracement = df['high'].iloc[level_idx:level_idx+forward_window].max()
                    retracement_ratio = (max_retracement - level.price) / (df['high'].iloc[level_idx-20:level_idx].max() - level.price)
                    
                    if retracement_ratio > self.mitigation_threshold:
                        # Mitigation block formed
                        mb_price = (level.price + max_retracement) / 2
                        
                        level_obj = SMCLevel(
                            level_type=SMCLevelType.MITIGATION_BLOCK,
                            price=mb_price,
                            strength=retracement_ratio,
                            confidence=min(1.0, retracement_ratio),
                            timestamp=df.index[level_idx + np.argmax(df['high'].iloc[level_idx:level_idx+forward_window])],
                            timeframe="30c",
                            touched_count=1
                        )
                        mitigation_blocks.append(level_obj)
        
        return mitigation_blocks
    
    def identify_premium_discount_zones(
        self,
        df: pd.DataFrame
    ) -> Tuple[List[SMCLevel], List[SMCLevel]]:
        """
        Identify premium and discount zones based on moving averages
        
        Premium zones: Price significantly above fair value
        Discount zones: Price significantly below fair value
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (premium_zones, discount_zones)
        """
        premium_zones = []
        discount_zones = []
        
        # Calculate moving averages
        ma_20 = df['close'].rolling(window=20).mean()
        ma_50 = df['close'].rolling(window=50).mean()
        
        # Calculate fair value (average of MAs)
        fair_value = (ma_20 + ma_50) / 2
        
        # Calculate deviation bands
        std_dev = df['close'].rolling(window=50).std()
        premium_threshold = fair_value + (std_dev * 1.5)
        discount_threshold = fair_value - (std_dev * 1.5)
        
        for i in range(50, len(df)):
            current_price = df['close'].iloc[i]
            current_premium = premium_threshold.iloc[i]
            current_discount = discount_threshold.iloc[i]
            
            # Check for premium zone
            if current_price > current_premium:
                deviation = (current_price - fair_value.iloc[i]) / fair_value.iloc[i]
                
                level = SMCLevel(
                    level_type=SMCLevelType.PREMIUM_ZONE,
                    price=current_price,
                    strength=deviation,
                    confidence=min(1.0, deviation / 0.03),  # Normalize to 3% premium
                    timestamp=df.index[i],
                    timeframe="50c",
                    touched_count=1
                )
                premium_zones.append(level)
            
            # Check for discount zone
            elif current_price < current_discount:
                deviation = (fair_value.iloc[i] - current_price) / fair_value.iloc[i]
                
                level = SMCLevel(
                    level_type=SMCLevelType.DISCOUNT_ZONE,
                    price=current_price,
                    strength=deviation,
                    confidence=min(1.0, deviation / 0.03),  # Normalize to 3% discount
                    timestamp=df.index[i],
                    timeframe="50c",
                    touched_count=1
                )
                discount_zones.append(level)
        
        return premium_zones, discount_zones
    
    def scan_all_levels(self, df: pd.DataFrame) -> List[SMCLevel]:
        """
        Scan for all SMC levels in the dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of all identified SMC levels
        """
        all_levels = []
        
        # 1. Liquidity zones
        lh, ll = self.identify_liquidity_zones(df)
        all_levels.extend(lh)
        all_levels.extend(ll)
        
        # 2. Order blocks
        bob, sob = self.identify_order_blocks(df)
        all_levels.extend(bob)
        all_levels.extend(sob)
        
        # 3. Structural levels
        sh, sl = self.identify_structural_levels(df)
        all_levels.extend(sh)
        all_levels.extend(sl)
        
        # 4. Premium/Discount zones
        pz, dz = self.identify_premium_discount_zones(df)
        all_levels.extend(pz)
        all_levels.extend(dz)
        
        # 5. Mitigation blocks (based on structural levels)
        mb = self.identify_mitigation_blocks(df, sh + sl)
        all_levels.extend(mb)
        
        # Store levels
        self.levels = all_levels
        self.historical_levels.extend(all_levels)
        
        return all_levels
    
    def generate_trade_setups(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> List[SMCTradeSetup]:
        """
        Generate trade setups based on SMC levels
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
            
        Returns:
            List of trade setups
        """
        setups = []
        
        if not self.levels:
            self.scan_all_levels(df)
        
        # Group levels by type
        level_groups = {}
        for level in self.levels:
            if level.level_type not in level_groups:
                level_groups[level.level_type] = []
            level_groups[level.level_type].append(level)
        
        # Generate setups for each level type
        setups.extend(self._generate_liquidity_setups(level_groups, current_price, df))
        setups.extend(self._generate_order_block_setups(level_groups, current_price, df))
        setups.extend(self._generate_structural_setups(level_groups, current_price, df))
        
        # Sort by setup score
        setups.sort(key=lambda x: x.setup_score, reverse=True)
        
        return setups
    
    def _generate_liquidity_setups(
        self,
        level_groups: Dict,
        current_price: float,
        df: pd.DataFrame
    ) -> List[SMCTradeSetup]:
        """Generate setups from liquidity levels"""
        setups = []
        
        # Liquidity High setups (Short)
        if SMCLevelType.LIQUIDITY_HIGH in level_groups:
            for level in level_groups[SMCLevelType.LIQUIDITY_HIGH]:
                if level.price > current_price and abs(level.price - current_price) / current_price < 0.05:
                    # Entry zone slightly below liquidity high
                    entry_upper = level.price
                    entry_lower = level.price * 0.995  # 0.5% below
                    
                    setup = SMCTradeSetup(
                        symbol=df.attrs.get('symbol', 'UNKNOWN'),
                        direction='SHORT',
                        entry_zone=(entry_lower, entry_upper),
                        stop_loss=level.price * 1.01,  # 1% above
                        take_profit_1=level.price * 0.98,  # 2% profit
                        take_profit_2=level.price * 0.96,  # 4% profit
                        take_profit_3=level.price * 0.94,  # 6% profit
                        confidence=level.confidence,
                        setup_score=level.strength * 10,
                        levels_used=[level],
                        timestamp=pd.Timestamp.now()
                    )
                    setups.append(setup)
        
        # Liquidity Low setups (Long)
        if SMCLevelType.LIQUIDITY_LOW in level_groups:
            for level in level_groups[SMCLevelType.LIQUIDITY_LOW]:
                if level.price < current_price and abs(level.price - current_price) / current_price < 0.05:
                    # Entry zone slightly above liquidity low
                    entry_lower = level.price
                    entry_upper = level.price * 1.005  # 0.5% above
                    
                    setup = SMCTradeSetup(
                        symbol=df.attrs.get('symbol', 'UNKNOWN'),
                        direction='LONG',
                        entry_zone=(entry_lower, entry_upper),
                        stop_loss=level.price * 0.99,  # 1% below
                        take_profit_1=level.price * 1.02,  # 2% profit
                        take_profit_2=level.price * 1.04,  # 4% profit
                        take_profit_3=level.price * 1.06,  # 6% profit
                        confidence=level.confidence,
                        setup_score=level.strength * 10,
                        levels_used=[level],
                        timestamp=pd.Timestamp.now()
                    )
                    setups.append(setup)
        
        return setups
    
    def _generate_order_block_setups(
        self,
        level_groups: Dict,
        current_price: float,
        df: pd.DataFrame
    ) -> List[SMCTradeSetup]:
        """Generate setups from order blocks"""
        setups = []
        
        # Bullish Order Block setups (Long)
        if SMCLevelType.ORDER_BLOCK_BULLISH in level_groups:
            for level in level_groups[SMCLevelType.ORDER_BLOCK_BULLISH]:
                if level.price < current_price and abs(level.price - current_price) / current_price < 0.03:
                    # Entry zone slightly above order block
                    entry_lower = level.price
                    entry_upper = level.price * 1.003  # 0.3% above
                    
                    setup = SMCTradeSetup(
                        symbol=df.attrs.get('symbol', 'UNKNOWN'),
                        direction='LONG',
                        entry_zone=(entry_lower, entry_upper),
                        stop_loss=level.price * 0.995,  # 0.5% below
                        take_profit_1=level.price * 1.015,  # 1.5% profit
                        take_profit_2=level.price * 1.03,   # 3% profit
                        take_profit_3=level.price * 1.045,  # 4.5% profit
                        confidence=level.confidence,
                        setup_score=level.strength * 8,
                        levels_used=[level],
                        timestamp=pd.Timestamp.now()
                    )
                    setups.append(setup)
        
        # Bearish Order Block setups (Short)
        if SMCLevelType.ORDER_BLOCK_BEARISH in level_groups:
            for level in level_groups[SMCLevelType.ORDER_BLOCK_BEARISH]:
                if level.price > current_price and abs(level.price - current_price) / current_price < 0.03:
                    # Entry zone slightly below order block
                    entry_upper = level.price
                    entry_lower = level.price * 0.997  # 0.3% below
                    
                    setup = SMCTradeSetup(
                        symbol=df.attrs.get('symbol', 'UNKNOWN'),
                        direction='SHORT',
                        entry_zone=(entry_lower, entry_upper),
                        stop_loss=level.price * 1.005,  # 0.5% above
                        take_profit_1=level.price * 0.985,  # 1.5% profit
                        take_profit_2=level.price * 0.97,   # 3% profit
                        take_profit_3=level.price * 0.955,  # 4.5% profit
                        confidence=level.confidence,
                        setup_score=level.strength * 8,
                        levels_used=[level],
                        timestamp=pd.Timestamp.now()
                    )
                    setups.append(setup)
        
        return setups
    
    def _generate_structural_setups(
        self,
        level_groups: Dict,
        current_price: float,
        df: pd.DataFrame
    ) -> List[SMCTradeSetup]:
        """Generate setups from structural levels"""
        setups = []
        
        # Structural Low setups (Long)
        if SMCLevelType.STRUCTURAL_LOW in level_groups:
            for level in level_groups[SMCLevelType.STRUCTURAL_LOW]:
                if level.price < current_price and abs(level.price - current_price) / current_price < 0.05:
                    # Entry zone near structural low
                    entry_lower = level.price * 0.998
                    entry_upper = level.price * 1.002
                    
                    setup = SMCTradeSetup(
                        symbol=df.attrs.get('symbol', 'UNKNOWN'),
                        direction='LONG',
                        entry_zone=(entry_lower, entry_upper),
                        stop_loss=level.price * 0.99,  # 1% below
                        take_profit_1=level.price * 1.02,  # 2% profit
                        take_profit_2=level.price * 1.04,   # 4% profit
                        take_profit_3=level.price * 1.06,  # 6% profit
                        confidence=level.confidence,
                        setup_score=level.strength * 7,
                        levels_used=[level],
                        timestamp=pd.Timestamp.now()
                    )
                    setups.append(setup)
        
        # Structural High setups (Short)
        if SMCLevelType.STRUCTURAL_HIGH in level_groups:
            for level in level_groups[SMCLevelType.STRUCTURAL_HIGH]:
                if level.price > current_price and abs(level.price - current_price) / current_price < 0.05:
                    # Entry zone near structural high
                    entry_lower = level.price * 0.998
                    entry_upper = level.price * 1.002
                    
                    setup = SMCTradeSetup(
                        symbol=df.attrs.get('symbol', 'UNKNOWN'),
                        direction='SHORT',
                        entry_zone=(entry_lower, entry_upper),
                        stop_loss=level.price * 1.01,  # 1% above
                        take_profit_1=level.price * 0.98,  # 2% profit
                        take_profit_2=level.price * 0.96,   # 4% profit
                        take_profit_3=level.price * 0.94,  # 6% profit
                        confidence=level.confidence,
                        setup_score=level.strength * 7,
                        levels_used=[level],
                        timestamp=pd.Timestamp.now()
                    )
                    setups.append(setup)
        
        return setups
    
    def update_levels_with_price_action(
        self,
        df: pd.DataFrame,
        current_price: float
    ):
        """
        Update existing levels based on recent price action
        
        Args:
            df: DataFrame with OHLCV data
            current_price: Current market price
        """
        updated_levels = []
        
        for level in self.levels:
            # Check if price has touched or broken the level
            touch_threshold = level.price * 0.005  # 0.5%
            
            if abs(current_price - level.price) <= touch_threshold:
                level.touched_count += 1
                level.last_touched = df.index[-1]
            
            # Check for break (more than 1% beyond level)
            break_threshold = level.price * 0.01
            
            if level.level_type in [SMCLevelType.LIQUIDITY_HIGH, SMCLevelType.STRUCTURAL_HIGH]:
                if current_price > (level.price + break_threshold):
                    level.broken_count += 1
            elif level.level_type in [SMCLevelType.LIQUIDITY_LOW, SMCLevelType.STRUCTURAL_LOW]:
                if current_price < (level.price - break_threshold):
                    level.broken_count += 1
            
            # Keep levels that are still relevant
            if level.touched_count > 0 or level.broken_count < 3:
                updated_levels.append(level)
        
        self.levels = updated_levels
    
    def get_level_statistics(self) -> Dict:
        """
        Get statistics about identified levels
        
        Returns:
            Dictionary with level statistics
        """
        stats = {
            'total_levels': len(self.levels),
            'level_types': {},
            'strength_distribution': {
                'weak': 0,      # 0.0 - 0.3
                'moderate': 0,  # 0.3 - 0.7
                'strong': 0     # 0.7 - 1.0
            },
            'confidence_distribution': {
                'low': 0,       # 0.0 - 0.3
                'medium': 0,   # 0.3 - 0.7
                'high': 0      # 0.7 - 1.0
            }
        }
        
        # Count level types
        for level in self.levels:
            level_type = level.level_type.value
            if level_type not in stats['level_types']:
                stats['level_types'][level_type] = 0
            stats['level_types'][level_type] += 1
            
            # Strength distribution
            if level.strength <= 0.3:
                stats['strength_distribution']['weak'] += 1
            elif level.strength <= 0.7:
                stats['strength_distribution']['moderate'] += 1
            else:
                stats['strength_distribution']['strong'] += 1
            
            # Confidence distribution
            if level.confidence <= 0.3:
                stats['confidence_distribution']['low'] += 1
            elif level.confidence <= 0.7:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['high'] += 1
        
        return stats

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 100)  # 0.1% mean return, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * np.random.uniform(0.995, 1.005, 100),
        'high': prices * np.random.uniform(1.001, 1.01, 100),
        'low': prices * np.random.uniform(0.99, 0.999, 100),
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, 100)
    }, index=dates)
    
    # Initialize SMC analyzer
    smc = SmartMoneyConcepts()
    
    print("=== SMART MONEY CONCEPTS ANALYSIS ===\n")
    
    # Scan for all levels
    levels = smc.scan_all_levels(df)
    print(f"Identified {len(levels)} SMC levels")
    
    # Get statistics
    stats = smc.get_level_statistics()
    print(f"\nLevel Statistics:")
    for level_type, count in stats['level_types'].items():
        print(f"  {level_type}: {count}")
    
    print(f"\nStrength Distribution:")
    for strength, count in stats['strength_distribution'].items():
        print(f"  {strength}: {count}")
    
    # Generate trade setups
    current_price = df['close'].iloc[-1]
    setups = smc.generate_trade_setups(df, current_price)
    
    print(f"\nGenerated {len(setups)} trade setups:")
    for i, setup in enumerate(setups[:3]):  # Show top 3
        print(f"\nSetup {i+1}:")
        print(f"  Direction: {setup.direction}")
        print(f"  Entry Zone: ${setup.entry_zone[0]:.2f} - ${setup.entry_zone[1]:.2f}")
        print(f"  Stop Loss: ${setup.stop_loss:.2f}")
        print(f"  TP1/TP2/TP3: ${setup.take_profit_1:.2f} / ${setup.take_profit_2:.2f} / ${setup.take_profit_3:.2f}")
        print(f"  Confidence: {setup.confidence:.2f}")
        print(f"  Setup Score: {setup.setup_score:.1f}")