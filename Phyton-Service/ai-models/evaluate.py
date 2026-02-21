"""
Evaluation & Backtesting for LSTM-GRU Hybrid Model
===================================================

Performance metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Sharpe Ratio
- Win Rate
- Profit/Loss simulation
- Trading statistics
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from lstm_gru_hybrid import create_model
from data_loader import FeatureDataLoader, create_dataloaders


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """

    def __init__(
        self,
        model_path: str = './checkpoints/lstm_gru_best.pth',
        input_dim: int = 150,
        device: str = None
    ):
        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model
        print(f"📥 Loading model from: {model_path}")
        self.model = create_model(input_dim=input_dim, device=self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✅ Model loaded successfully\n")

        # Class names
        self.class_names = ['BUY', 'SELL', 'HOLD']

    def evaluate_test_set(self, test_loader) -> Dict:
        """
        Evaluate model on test set

        Returns comprehensive metrics
        """
        print(f"{'='*60}")
        print(f"📊 EVALUATING MODEL ON TEST SET")
        print(f"{'='*60}\n")

        all_predictions = []
        all_labels = []
        all_probabilities = []

        # Get predictions
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits, _ = self.model(X_batch)
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, labels=[0, 1, 2]
        )

        # Weighted metrics
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])

        # Classification report
        report = classification_report(
            all_labels, all_predictions,
            target_names=self.class_names,
            output_dict=True
        )

        metrics = {
            'accuracy': float(accuracy),
            'precision': {
                'BUY': float(precision[0]),
                'SELL': float(precision[1]),
                'HOLD': float(precision[2]),
                'weighted': float(precision_weighted)
            },
            'recall': {
                'BUY': float(recall[0]),
                'SELL': float(recall[1]),
                'HOLD': float(recall[2]),
                'weighted': float(recall_weighted)
            },
            'f1_score': {
                'BUY': float(f1[0]),
                'SELL': float(f1[1]),
                'HOLD': float(f1[2]),
                'weighted': float(f1_weighted)
            },
            'support': {
                'BUY': int(support[0]),
                'SELL': int(support[1]),
                'HOLD': int(support[2]),
                'total': int(support.sum())
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        # Print results
        self._print_metrics(metrics)

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Print formatted metrics"""
        print(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%\n")

        print(f"Per-class Metrics:")
        print(f"{'Class':<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print(f"{'-'*60}")
        for class_name in self.class_names:
            print(f"{class_name:<6} "
                  f"{metrics['precision'][class_name]*100:>10.2f}% "
                  f"{metrics['recall'][class_name]*100:>10.2f}% "
                  f"{metrics['f1_score'][class_name]*100:>10.2f}% "
                  f"{metrics['support'][class_name]:>10d}")

        print(f"\nWeighted Averages:")
        print(f"  Precision: {metrics['precision']['weighted']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']['weighted']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1_score']['weighted']*100:.2f}%")

        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(f"{'':>10} {'BUY':>8} {'SELL':>8} {'HOLD':>8}")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:>10} {cm[i][0]:>8d} {cm[i][1]:>8d} {cm[i][2]:>8d}")

    def backtest(
        self,
        test_loader,
        initial_capital: float = 10000.0,
        position_size: float = 0.1,  # 10% of capital per trade
        commission: float = 0.001  # 0.1% commission
    ) -> Dict:
        """
        Backtest trading strategy

        Simulates trading based on model predictions

        Args:
            test_loader: Test data loader
            initial_capital: Starting capital ($)
            position_size: Fraction of capital to use per trade
            commission: Trading commission (fraction)

        Returns:
            backtest_results: Trading statistics
        """
        print(f"\n{'='*60}")
        print(f"📈 BACKTESTING TRADING STRATEGY")
        print(f"{'='*60}\n")

        capital = initial_capital
        position = None  # 'long' or 'short' or None
        entry_price = 0.0
        trades = []

        # Get predictions
        all_predictions = []
        all_prices = []  # We'll need prices for backtesting

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits, _ = self.model(X_batch)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())

                # Extract prices from features
                # Assuming 'close' is the first feature
                # (This is simplified - in real scenario, need actual price data)
                prices = X_batch[:, -1, 0].cpu().numpy()  # Last timestep, first feature
                all_prices.extend(prices)

        all_predictions = np.array(all_predictions)
        all_prices = np.array(all_prices)

        # Simulate trading
        for i in range(len(all_predictions)):
            signal = all_predictions[i]
            price = all_prices[i]

            # BUY signal
            if signal == 0 and position != 'long':
                # Close short position if any
                if position == 'short':
                    profit = (entry_price - price) * (capital * position_size) / entry_price
                    profit -= abs(profit) * commission
                    capital += profit
                    trades.append({
                        'type': 'close_short',
                        'price': price,
                        'profit': profit,
                        'capital': capital
                    })

                # Open long position
                entry_price = price
                position = 'long'
                trades.append({
                    'type': 'open_long',
                    'price': price,
                    'profit': 0.0,
                    'capital': capital
                })

            # SELL signal
            elif signal == 1 and position != 'short':
                # Close long position if any
                if position == 'long':
                    profit = (price - entry_price) * (capital * position_size) / entry_price
                    profit -= abs(profit) * commission
                    capital += profit
                    trades.append({
                        'type': 'close_long',
                        'price': price,
                        'profit': profit,
                        'capital': capital
                    })

                # Open short position
                entry_price = price
                position = 'short'
                trades.append({
                    'type': 'open_short',
                    'price': price,
                    'profit': 0.0,
                    'capital': capital
                })

            # HOLD signal - do nothing

        # Close any remaining position
        if position is not None:
            final_price = all_prices[-1]
            if position == 'long':
                profit = (final_price - entry_price) * (capital * position_size) / entry_price
                profit -= abs(profit) * commission
            else:  # short
                profit = (entry_price - final_price) * (capital * position_size) / entry_price
                profit -= abs(profit) * commission

            capital += profit
            trades.append({
                'type': f'close_{position}',
                'price': final_price,
                'profit': profit,
                'capital': capital
            })

        # Calculate statistics
        total_return = (capital - initial_capital) / initial_capital
        num_trades = len([t for t in trades if 'close' in t['type']])
        winning_trades = [t for t in trades if 'close' in t['type'] and t['profit'] > 0]
        losing_trades = [t for t in trades if 'close' in t['type'] and t['profit'] < 0]

        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0.0

        # Sharpe Ratio (simplified)
        returns = np.array([t['profit'] / initial_capital for t in trades if 'close' in t['type']])
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0

        results = {
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': num_trades,
            'num_winning_trades': len(winning_trades),
            'num_losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades
        }

        # Print results
        self._print_backtest_results(results)

        return results

    def _print_backtest_results(self, results: Dict):
        """Print formatted backtest results"""
        print(f"Initial Capital:  ${results['initial_capital']:,.2f}")
        print(f"Final Capital:    ${results['final_capital']:,.2f}")
        print(f"Total Return:     {results['total_return_pct']:+.2f}%")
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:   {results['num_trades']}")
        print(f"  Winning Trades: {results['num_winning_trades']}")
        print(f"  Losing Trades:  {results['num_losing_trades']}")
        print(f"  Win Rate:       {results['win_rate_pct']:.2f}%")
        print(f"\nProfit/Loss:")
        print(f"  Avg Win:        ${results['avg_win']:,.2f}")
        print(f"  Avg Loss:       ${results['avg_loss']:,.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:   {results['sharpe_ratio']:.2f}")

    def plot_confusion_matrix(self, metrics: Dict, save_path: str = './checkpoints/confusion_matrix.png'):
        """
        Plot confusion matrix

        Args:
            metrics: Metrics dict with confusion_matrix
            save_path: Path to save plot
        """
        cm = np.array(metrics['confusion_matrix'])

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"\n✅ Confusion matrix plot saved to: {save_path}")


def main():
    """
    Main evaluation pipeline
    """
    print(f"{'='*60}")
    print(f"🔬 MODEL EVALUATION & BACKTESTING")
    print(f"{'='*60}\n")

    # Configuration
    CONFIG = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 1000,
        'seq_length': 60,
        'batch_size': 32,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Load data
    print("📥 Loading test data...")
    data_loader = FeatureDataLoader(
        feature_service_url="http://localhost:5006",
        seq_length=CONFIG['seq_length'],
        device=CONFIG['device']
    )

    _, _, test_dataset = data_loader.prepare_data(
        symbol=CONFIG['symbol'],
        interval=CONFIG['interval'],
        limit=CONFIG['limit']
    )

    _, _, test_loader = create_dataloaders(
        None, None, test_dataset,
        batch_size=CONFIG['batch_size']
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path='./checkpoints/lstm_gru_best.pth',
        input_dim=data_loader.num_features,
        device=CONFIG['device']
    )

    # Evaluate
    metrics = evaluator.evaluate_test_set(test_loader)

    # Save metrics
    metrics_path = Path('./checkpoints/evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_path}")

    # Plot confusion matrix
    evaluator.plot_confusion_matrix(metrics)

    # Backtest
    backtest_results = evaluator.backtest(test_loader)

    # Save backtest results
    backtest_path = Path('./checkpoints/backtest_results.json')
    # Remove trades list for file size
    backtest_summary = {k: v for k, v in backtest_results.items() if k != 'trades'}
    with open(backtest_path, 'w') as f:
        json.dump(backtest_summary, f, indent=2)
    print(f"\n✅ Backtest results saved to: {backtest_path}")

    print(f"\n{'='*60}")
    print(f"🎉 EVALUATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
