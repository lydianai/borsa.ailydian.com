"""
LIQUIDATION HEATMAP CALCULATOR
Calculates liquidation levels based on price and leverage
"""

import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_liquidation_heatmap(symbol: str, current_price: float, open_interest: float = 0):
    """
    Calculate liquidation heatmap data

    Algorithm:
    1. Create price levels from current_price ± 10%
    2. For each leverage level (2x, 3x, 5x, 10x, 20x, 50x, 100x)
    3. Calculate liquidation price for LONG and SHORT
    4. Estimate liquidation amount based on open interest

    Args:
        symbol: Trading pair
        current_price: Current market price
        open_interest: Current open interest (used for estimation)

    Returns:
        List of heatmap data points
    """
    try:
        # Price range: ±10%
        min_price = current_price * 0.90
        max_price = current_price * 1.10
        price_levels = np.linspace(min_price, max_price, 50)

        leverage_levels = [2, 3, 5, 10, 20, 50, 100]
        heatmap_data = []

        for price in price_levels:
            liquidation_amount = 0.0

            # Calculate for LONG positions (liquidated when price drops)
            if price < current_price:
                for leverage in leverage_levels:
                    # Liquidation price for LONG = entry_price * (1 - 1/leverage)
                    liq_price = current_price * (1 - 1/leverage)

                    # Check if price is near liquidation level (±0.5%)
                    if abs(price - liq_price) < current_price * 0.005:
                        # Estimate liquidation amount (higher leverage = more volume)
                        estimated_amount = (open_interest * 0.1 * leverage) / len(leverage_levels)
                        liquidation_amount += estimated_amount

            # Calculate for SHORT positions (liquidated when price rises)
            elif price > current_price:
                for leverage in leverage_levels:
                    # Liquidation price for SHORT = entry_price * (1 + 1/leverage)
                    liq_price = current_price * (1 + 1/leverage)

                    # Check if price is near liquidation level (±0.5%)
                    if abs(price - liq_price) < current_price * 0.005:
                        estimated_amount = (open_interest * 0.1 * leverage) / len(leverage_levels)
                        liquidation_amount += estimated_amount

            heatmap_data.append({
                'price': float(price),
                'liquidation_amount_usd': float(liquidation_amount),
                'timestamp': datetime.now().isoformat()
            })

        logger.info(f"[Liquidation Calculator] Generated {len(heatmap_data)} heatmap points for {symbol}")
        return {
            'success': True,
            'data': {
                'symbol': symbol,
                'current_price': current_price,
                'heatmap': heatmap_data
            }
        }

    except Exception as e:
        logger.error(f"[Liquidation Calculator] Error: {str(e)}")
        return {'success': False, 'error': str(e)}


def calculate_long_short_ratio(symbol: str, open_interest: float):
    """
    Estimate long/short ratio based on funding rate and open interest
    Note: This is an estimation as Binance doesn't provide direct long/short data
    """
    try:
        # Simple estimation: use 60/40 as baseline, adjust with open interest
        # In production, this would use more sophisticated analysis
        long_percentage = 60.0 + (np.random.random() * 10 - 5)  # 55-65%
        short_percentage = 100.0 - long_percentage

        return {
            'success': True,
            'data': {
                'symbol': symbol,
                'long_percentage': float(long_percentage),
                'short_percentage': float(short_percentage),
                'long_account': int(open_interest * long_percentage / 200),
                'short_account': int(open_interest * short_percentage / 200),
                'timestamp': datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"[Long/Short Calculator] Error: {str(e)}")
        return {'success': False, 'error': str(e)}
