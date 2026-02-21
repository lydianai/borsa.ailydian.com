"""
CONSENSUS ENGINE
Aggregates predictions from 14 AI models into unified signals
White-hat compliant - transparent voting system
"""

import numpy as np
from typing import List, Dict, Any
from datetime import datetime


class ConsensusEngine:
    """
    Professional consensus engine for multi-model aggregation

    Features:
    - Weighted voting based on model confidence
    - Risk-reward calculation
    - Signal quality scoring
    - Transparent decision making
    """

    def __init__(self):
        # Model weights (can be adjusted based on backtesting)
        self.model_weights = {
            'lstm_standard': 1.0,
            'lstm_bidirectional': 1.1,
            'lstm_stacked': 1.2,
            'gru_standard': 1.0,
            'gru_bidirectional': 1.1,
            'gru_stacked': 1.2,
            'gru_attention': 1.3,
            'gru_residual': 1.2,
            'transformer_standard': 1.4,
            'transformer_timeseries': 1.4,
            'transformer_informer': 1.3,
            'xgboost': 1.1,
            'lightgbm': 1.1,
            'catboost': 1.1,
        }

    def aggregate_predictions(
        self,
        predictions: List[Dict[str, Any]],
        current_price: float
    ) -> Dict[str, Any]:
        """
        Aggregate multiple model predictions into consensus signal

        Args:
            predictions: List of {model, action, confidence, prediction}
            current_price: Current market price

        Returns:
            Consensus signal with confidence and targets
        """
        if not predictions:
            return self._no_signal()

        # Separate predictions by action
        buy_votes = []
        sell_votes = []
        hold_votes = []

        for pred in predictions:
            model = pred['model_name']
            action = pred['action']
            confidence = pred['confidence']

            weight = self.model_weights.get(model, 1.0)
            weighted_confidence = confidence * weight

            if action == 'BUY':
                buy_votes.append(weighted_confidence)
            elif action == 'SELL':
                sell_votes.append(weighted_confidence)
            else:
                hold_votes.append(weighted_confidence)

        # Calculate consensus
        total_models = len(predictions)
        buy_strength = sum(buy_votes) / total_models if buy_votes else 0
        sell_strength = sum(sell_votes) / total_models if sell_votes else 0
        hold_strength = sum(hold_votes) / total_models if hold_votes else 0

        # Determine action
        strengths = {
            'BUY': buy_strength,
            'SELL': sell_strength,
            'HOLD': hold_strength
        }

        action = max(strengths, key=strengths.get)
        confidence = strengths[action]

        # Calculate targets
        entry_price = current_price

        if action == 'BUY':
            target_price = entry_price * 1.02  # 2% profit target
            stop_loss = entry_price * 0.99     # 1% stop loss
        elif action == 'SELL':
            target_price = entry_price * 0.98  # 2% profit target
            stop_loss = entry_price * 1.01     # 1% stop loss
        else:  # HOLD
            target_price = entry_price
            stop_loss = entry_price * 0.995

        # Risk-reward ratio
        if action != 'HOLD':
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = 0

        # Signal quality
        quality = self._calculate_quality(
            confidence=confidence,
            agreement=len(buy_votes if action == 'BUY' else sell_votes) / total_models,
            risk_reward=risk_reward
        )

        return {
            'action': action,
            'confidence': float(confidence * 100),  # Convert to percentage
            'entry_price': float(entry_price),
            'target_price': float(target_price),
            'stop_loss': float(stop_loss),
            'risk_reward': float(risk_reward),
            'quality': quality,
            'consensus': f"{len(buy_votes if action == 'BUY' else sell_votes if action == 'SELL' else hold_votes)}/{total_models} models",
            'model_breakdown': {
                'buy_votes': len(buy_votes),
                'sell_votes': len(sell_votes),
                'hold_votes': len(hold_votes),
                'buy_strength': float(buy_strength),
                'sell_strength': float(sell_strength),
                'hold_strength': float(hold_strength),
            },
            'timestamp': datetime.now().isoformat()
        }

    def _calculate_quality(
        self,
        confidence: float,
        agreement: float,
        risk_reward: float
    ) -> str:
        """Calculate signal quality grade"""

        # Weighted score
        score = (
            confidence * 0.4 +
            agreement * 0.4 +
            min(risk_reward / 3.0, 1.0) * 0.2
        )

        if score >= 0.8:
            return 'EXCELLENT'
        elif score >= 0.7:
            return 'GOOD'
        elif score >= 0.6:
            return 'FAIR'
        else:
            return 'POOR'

    def _no_signal(self) -> Dict[str, Any]:
        """Return no signal structure"""
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'entry_price': 0.0,
            'target_price': 0.0,
            'stop_loss': 0.0,
            'risk_reward': 0.0,
            'quality': 'NO_DATA',
            'consensus': '0/0 models',
            'model_breakdown': {},
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Test consensus engine
    engine = ConsensusEngine()

    # Mock predictions
    test_predictions = [
        {'model_name': 'lstm_standard', 'action': 'BUY', 'confidence': 0.75, 'prediction': 0.75},
        {'model_name': 'gru_attention', 'action': 'BUY', 'confidence': 0.82, 'prediction': 0.82},
        {'model_name': 'transformer_standard', 'action': 'BUY', 'confidence': 0.88, 'prediction': 0.88},
        {'model_name': 'xgboost', 'action': 'HOLD', 'confidence': 0.55, 'prediction': 0.55},
    ]

    result = engine.aggregate_predictions(test_predictions, 67000.0)

    print("=" * 60)
    print("CONSENSUS ENGINE TEST")
    print("=" * 60)
    print(f"Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Entry: ${result['entry_price']:,.2f}")
    print(f"Target: ${result['target_price']:,.2f}")
    print(f"Stop Loss: ${result['stop_loss']:,.2f}")
    print(f"Risk:Reward = 1:{result['risk_reward']:.2f}")
    print(f"Quality: {result['quality']}")
    print(f"Consensus: {result['consensus']}")
