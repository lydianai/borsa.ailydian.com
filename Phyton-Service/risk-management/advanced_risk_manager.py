"""
ADVANCED RISK MANAGEMENT SYSTEM
===============================

Premium risk management with Kelly Criterion, 
dynamic position sizing, and adaptive stop-loss

Features:
- Kelly Criterion position sizing
- Dynamic volatility-adjusted stops
- Correlation-adjusted portfolio risk
- Drawdown protection
- Market regime detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

class MarketRegime(Enum):
    """Market condition regimes"""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    STRESSED = "stressed"

@dataclass
class Position:
    """Position details"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    size: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    reward_amount: float
    confidence: float  # 0.0 - 1.0

@dataclass
class RiskMetrics:
    """Risk metrics for portfolio"""
    portfolio_value: float
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    win_rate: float
    risk_exposure: float
    correlation_adjusted_risk: float

class AdvancedRiskManager:
    """
    Premium Risk Management System
    
    Implements sophisticated risk management techniques:
    - Kelly Criterion for optimal position sizing
    - Volatility-adjusted stop losses
    - Market regime adaptation
    - Correlation-aware portfolio risk
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_portfolio_risk: float = 0.02,  # 2% portfolio risk per trade
        max_correlated_risk: float = 0.05,  # 5% correlated asset group risk
        kelly_fraction: float = 0.25,  # Quarter Kelly to reduce variance
        max_leverage: float = 2.0,
        drawdown_limit: float = 0.20  # 20% maximum drawdown
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_risk = max_correlated_risk
        self.kelly_fraction = kelly_fraction
        self.max_leverage = max_leverage
        self.drawdown_limit = drawdown_limit
        
        # Portfolio tracking
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.capital_history: List[float] = [initial_capital]
        self.peak_capital = initial_capital
        
        # Market state
        self.volatility_regime = 1.0  # 1.0 = normal, >1.0 = high volatility
        self.market_regime = MarketRegime.TRENDING
        self.correlation_matrix = np.eye(5)  # Placeholder for 5 assets
        
        # Risk limits
        self.daily_loss_limit = initial_capital * 0.03  # 3% daily loss limit
        self.daily_losses = 0.0
        self.max_consecutive_losses = 3
        self.consecutive_losses = 0
        
    def calculate_kelly_position_size(
        self,
        win_rate: float,
        win_loss_ratio: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        f* = p - (1-p)/(b)
        where:
        f* = fraction of capital to bet
        p = probability of winning
        b = average win/average loss
        
        Args:
            win_rate: Historical win rate (0.0 - 1.0)
            win_loss_ratio: Average win / average loss
            confidence: Model confidence adjustment (0.0 - 1.0)
            
        Returns:
            Optimal fraction of capital to risk
        """
        if win_rate <= 0 or win_loss_ratio <= 0:
            return 0.0
            
        # Kelly formula
        kelly_fraction = win_rate - (1 - win_rate) / win_loss_ratio
        
        # Apply confidence adjustment
        adjusted_kelly = kelly_fraction * confidence
        
        # Use fractional Kelly to reduce variance
        fractional_kelly = adjusted_kelly * self.kelly_fraction
        
        # Ensure positive and reasonable values
        return max(0.0, min(fractional_kelly, self.max_portfolio_risk))
    
    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        volatility_multiplier: float = 2.0
    ) -> float:
        """
        Calculate dynamic stop-loss based on volatility
        
        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            atr: Average True Range
            volatility_multiplier: ATR multiplier for stop distance
            
        Returns:
            Stop-loss price
        """
        # Base stop distance
        stop_distance = atr * volatility_multiplier * self.volatility_regime
        
        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance
            
        return stop_loss
    
    def calculate_take_profit(
        self,
        entry_price: float,
        direction: str,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take-profit level based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            direction: 'LONG' or 'SHORT'
            stop_loss: Stop-loss price
            risk_reward_ratio: Target risk-reward ratio
            
        Returns:
            Take-profit price
        """
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = risk_amount * risk_reward_ratio
        
        if direction == 'LONG':
            take_profit = entry_price + reward_amount
        else:  # SHORT
            take_profit = entry_price - reward_amount
            
        return take_profit
    
    def assess_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """
        Assess current market regime using statistical measures
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            Market regime classification
        """
        if len(price_data) < 50:
            return MarketRegime.TRENDING
            
        # Calculate returns
        returns = price_data['close'].pct_change().dropna()
        
        # Volatility (standard deviation of returns)
        volatility = returns.std()
        
        # Trend strength (linear regression slope)
        x = np.arange(len(returns))
        slope = np.polyfit(x, returns.cumsum(), 1)[0]
        trend_strength = abs(slope)
        
        # Autocorrelation
        autocorr = returns.autocorr(lag=1)
        
        # Regime classification
        if volatility > returns.std() * 1.5:
            return MarketRegime.VOLATILE
        elif trend_strength > 0.001 and abs(autocorr) > 0.1:
            return MarketRegime.TRENDING
        elif volatility < returns.std() * 0.7:
            return MarketRegime.RANGING
        else:
            return MarketRegime.STRESSED
    
    def adjust_for_correlation_risk(
        self,
        symbol: str,
        proposed_position_size: float,
        correlation_group: List[str]
    ) -> float:
        """
        Adjust position size based on correlation with existing positions
        
        Args:
            symbol: Asset symbol
            proposed_position_size: Initially calculated position size
            correlation_group: Symbols in the same correlation group
            
        Returns:
            Adjusted position size considering correlation risk
        """
        # Calculate current exposure to correlation group
        group_exposure = 0.0
        for position in self.positions:
            if position.symbol in correlation_group:
                group_exposure += position.risk_amount
                
        # If group exposure is high, reduce new position
        if group_exposure > self.current_capital * self.max_correlated_risk:
            reduction_factor = max(0.1, 1.0 - (group_exposure / 
                                            (self.current_capital * self.max_correlated_risk)))
            return proposed_position_size * reduction_factor
            
        return proposed_position_size
    
    def calculate_position_metrics(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float,
        atr: float,
        correlation_group: List[str]
    ) -> Dict:
        """
        Calculate all position metrics and risk parameters
        
        Args:
            symbol: Asset symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            confidence: Model confidence (0.0 - 1.0)
            atr: Average True Range
            correlation_group: Symbols correlated with this asset
            
        Returns:
            Dictionary with position metrics
        """
        # Win/loss ratio
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 1.0
        
        # Kelly position size
        kelly_size = self.calculate_kelly_position_size(win_rate, win_loss_ratio, confidence)
        
        # Dynamic stop loss
        stop_loss = self.calculate_dynamic_stop_loss(entry_price, direction, atr)
        
        # Take profit
        take_profit = self.calculate_take_profit(entry_price, direction, stop_loss)
        
        # Risk/reward amounts
        risk_amount = abs(entry_price - stop_loss)
        reward_amount = abs(take_profit - entry_price)
        
        # Position size in currency terms
        max_risk_currency = self.current_capital * self.max_portfolio_risk
        position_size_currency = min(max_risk_currency, 
                                   kelly_size * self.current_capital)
        
        # Adjust for correlation risk
        adjusted_size = self.adjust_for_correlation_risk(
            symbol, position_size_currency, correlation_group
        )
        
        # Final position size
        final_position_size = adjusted_size / entry_price
        
        return {
            'position_size': final_position_size,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': reward_amount / risk_amount if risk_amount > 0 else 0,
            'kelly_fraction': kelly_size,
            'adjusted_kelly_fraction': adjusted_size / self.current_capital
        }
    
    def can_enter_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        confidence: float
    ) -> Tuple[bool, str]:
        """
        Check if position can be entered based on risk rules
        
        Args:
            symbol: Asset symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            confidence: Model confidence
            
        Returns:
            Tuple of (can_enter, reason)
        """
        # Check drawdown limit
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.drawdown_limit:
            return False, f"Drawdown limit exceeded ({current_drawdown:.1%})"
        
        # Check daily loss limit
        if self.daily_losses > self.daily_loss_limit:
            return False, f"Daily loss limit exceeded ({self.daily_losses:.2f})"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"Max consecutive losses reached ({self.consecutive_losses})"
        
        # Check leverage limit
        total_exposure = sum(pos.size * pos.entry_price for pos in self.positions)
        potential_exposure = total_exposure + (self.current_capital * 0.01)  # Estimate
        leverage = potential_exposure / self.current_capital
        
        if leverage > self.max_leverage:
            return False, f"Leverage limit exceeded ({leverage:.2f}x)"
        
        return True, "OK"
    
    def execute_trade(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        confidence: float
    ) -> bool:
        """
        Execute trade and update risk management state
        
        Args:
            symbol: Asset symbol
            direction: 'LONG' or 'SHORT'
            entry_price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Model confidence
            
        Returns:
            True if trade executed successfully
        """
        # Create position
        risk_amount = abs(entry_price - stop_loss) * size
        reward_amount = abs(take_profit - entry_price) * size
        
        position = Position(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            confidence=confidence
        )
        
        # Add to positions
        self.positions.append(position)
        
        # Update capital (assuming execution)
        self.current_capital -= risk_amount  # Risk amount reserved
        self.capital_history.append(self.current_capital)
        
        # Update peak capital
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        return True
    
    def update_position(
        self,
        symbol: str,
        exit_price: float,
        exit_type: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'MANUAL'
    ) -> Dict:
        """
        Update position on exit and calculate P&L
        
        Args:
            symbol: Asset symbol
            exit_price: Exit price
            exit_type: Type of exit
            
        Returns:
            Trade results dictionary
        """
        # Find position
        position = None
        position_index = None
        
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                position = pos
                position_index = i
                break
        
        if position is None:
            return {'success': False, 'error': 'Position not found'}
        
        # Calculate P&L
        if position.direction == 'LONG':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.size
        
        # Update capital
        self.current_capital += position.risk_amount + pnl  # Return risk + P&L
        self.capital_history.append(self.current_capital)
        
        # Update peak capital
        self.peak_capital = max(self.peak_capital, self.current_capital)
        
        # Update trade history
        trade_record = {
            'symbol': symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': pnl,
            'exit_type': exit_type,
            'confidence': position.confidence,
            'timestamp': pd.Timestamp.now()
        }
        
        self.trade_history.append(trade_record)
        
        # Update consecutive losses counter
        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_losses += abs(pnl)
        else:
            self.consecutive_losses = 0
        
        # Remove position
        if position_index is not None:
            self.positions.pop(position_index)
        
        return {
            'success': True,
            'pnl': pnl,
            'trade_record': trade_record
        }
    
    def get_portfolio_risk_metrics(self) -> RiskMetrics:
        """
        Calculate current portfolio risk metrics
        
        Returns:
            RiskMetrics dataclass
        """
        if len(self.capital_history) < 2:
            returns = np.array([0.0])
        else:
            capital_array = np.array(self.capital_history)
            returns = np.diff(capital_array) / capital_array[:-1]
        
        # Calculate metrics
        portfolio_value = self.current_capital
        max_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 and np.std(returns) > 0 else 0.0
        
        # Win rate
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0.0
        
        # Risk exposure
        total_risk = sum(pos.risk_amount for pos in self.positions)
        risk_exposure = total_risk / self.current_capital if self.current_capital > 0 else 0.0
        
        # Correlation adjusted risk (simplified)
        correlation_adjusted_risk = risk_exposure * np.mean(np.abs(self.correlation_matrix))
        
        return RiskMetrics(
            portfolio_value=portfolio_value,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            risk_exposure=risk_exposure,
            correlation_adjusted_risk=correlation_adjusted_risk
        )
    
    def update_volatility_regime(self, recent_atr_values: List[float]):
        """
        Update volatility regime based on recent ATR values
        
        Args:
            recent_atr_values: List of recent ATR values
        """
        if not recent_atr_values:
            return
            
        current_atr = np.mean(recent_atr_values[-20:])  # 20-period average
        historical_atr = np.mean(recent_atr_values)  # All available history
        
        if historical_atr > 0:
            self.volatility_regime = current_atr / historical_atr
        else:
            self.volatility_regime = 1.0
    
    def reset_daily_counters(self):
        """Reset daily loss counters"""
        self.daily_losses = 0.0

# Example usage
if __name__ == "__main__":
    # Initialize risk manager
    rm = AdvancedRiskManager(
        initial_capital=10000.0,
        max_portfolio_risk=0.02,  # 2% per trade
        kelly_fraction=0.25,      # Quarter Kelly
        drawdown_limit=0.20       # 20% max drawdown
    )
    
    # Example trade calculation
    metrics = rm.calculate_position_metrics(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=45000.0,
        win_rate=0.65,
        avg_win=1200.0,
        avg_loss=-600.0,
        confidence=0.85,
        atr=800.0,
        correlation_group=['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    )
    
    print("=== ADVANCED RISK MANAGEMENT EXAMPLE ===")
    print(f"Position Size: {metrics['position_size']:.6f}")
    print(f"Stop Loss: ${metrics['stop_loss']:.2f}")
    print(f"Take Profit: ${metrics['take_profit']:.2f}")
    print(f"Risk/Reward Ratio: {metrics['risk_reward_ratio']:.2f}")
    print(f"Kelly Fraction: {metrics['kelly_fraction']:.4f}")
    print(f"Adjusted Kelly: {metrics['adjusted_kelly_fraction']:.4f}")
    
    # Check if we can enter position
    can_enter, reason = rm.can_enter_position(
        symbol='BTCUSDT',
        direction='LONG',
        entry_price=45000.0,
        confidence=0.85
    )
    
    print(f"\nCan Enter Position: {can_enter}")
    print(f"Reason: {reason}")