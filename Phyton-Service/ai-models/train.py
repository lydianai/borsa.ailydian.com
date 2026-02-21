"""
Training Script for LSTM-GRU Hybrid Model
==========================================

Full training pipeline:
- Data loading from Feature Engineering service
- Model initialization
- Training loop with validation
- Model checkpointing
- Metrics tracking
- Early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple
import sys

from lstm_gru_hybrid import LSTMGRUHybrid, create_model, count_parameters
from data_loader import FeatureDataLoader, create_dataloaders


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    """

    def __init__(self, weights: Dict[int, float], device: str = 'cpu'):
        super().__init__()
        # Convert weights dict to tensor
        weight_list = [weights.get(i, 1.0) for i in range(3)]
        self.weight = torch.FloatTensor(weight_list).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, labels)


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"   EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        if self.verbose:
            print(f"   ✅ Validation loss decreased ({self.val_loss_min:.4f} → {val_loss:.4f}). Saving model...")
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class Trainer:
    """
    Model Trainer
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        class_weights: Dict[int, float],
        device: str = 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        # Loss function (weighted)
        self.criterion = WeightedCrossEntropyLoss(class_weights, device=device)

        # Optimizer (AdamW with weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )

        # Early stopping
        self.early_stopping = EarlyStopping(patience=10, verbose=True)

        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Metrics history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch

        Returns:
            average_loss, accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):
            # Forward pass
            logits, _ = self.model(X_batch)

            # Calculate loss
            loss = self.criterion(logits, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate model

        Returns:
            average_loss, accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, epochs: int = 100) -> Dict:
        """
        Full training loop

        Args:
            epochs: Number of epochs

        Returns:
            history: Training metrics
        """
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING LSTM-GRU HYBRID MODEL")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Device: {self.device}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")

        best_checkpoint = self.checkpoint_dir / 'lstm_gru_best.pth'

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)

            # Print metrics
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                  f"LR: {current_lr:.6f}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Early stopping
            self.early_stopping(val_loss, self.model, str(best_checkpoint))
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break

        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE")
        print(f"{'='*60}\n")

        # Load best model
        self.model.load_state_dict(torch.load(best_checkpoint))

        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Training history saved to: {history_path}")

        return self.history

    def evaluate(self) -> Dict:
        """
        Evaluate model on test set

        Returns:
            metrics: Test metrics
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                logits, _ = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        # Calculate metrics
        accuracy = np.mean(all_predictions == all_labels)
        avg_loss = total_loss / len(self.test_loader)

        # Per-class accuracy
        class_names = ['BUY', 'SELL', 'HOLD']
        class_accuracies = {}
        for i, class_name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                class_acc = np.mean(all_predictions[mask] == all_labels[mask])
                class_accuracies[class_name] = class_acc

        metrics = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': len(all_labels),
            'label_distribution': {
                'BUY': int(np.sum(all_labels == 0)),
                'SELL': int(np.sum(all_labels == 1)),
                'HOLD': int(np.sum(all_labels == 2))
            }
        }

        print(f"\n{'='*60}")
        print(f"📊 TEST RESULTS")
        print(f"{'='*60}")
        print(f"Test Loss: {avg_loss:.4f}")
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        print(f"\nPer-class Accuracy:")
        for class_name, acc in class_accuracies.items():
            print(f"  {class_name:5s}: {acc*100:.2f}%")
        print(f"{'='*60}\n")

        return metrics


def main():
    """
    Main training pipeline
    """
    # Configuration
    CONFIG = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 1000,
        'seq_length': 60,
        'look_ahead': 5,
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"{'='*60}")
    print(f"⚙️  CONFIGURATION")
    print(f"{'='*60}")
    for key, value in CONFIG.items():
        print(f"{key:20s}: {value}")
    print(f"{'='*60}\n")

    # 1. Load data
    print("📥 Step 1: Loading data...")
    data_loader = FeatureDataLoader(
        feature_service_url="http://localhost:5006",
        seq_length=CONFIG['seq_length'],
        look_ahead=CONFIG['look_ahead'],
        device=CONFIG['device']
    )

    train_dataset, val_dataset, test_dataset = data_loader.prepare_data(
        symbol=CONFIG['symbol'],
        interval=CONFIG['interval'],
        limit=CONFIG['limit']
    )

    # Calculate class weights
    all_labels = np.concatenate([
        train_dataset.labels.cpu().numpy(),
        val_dataset.labels.cpu().numpy(),
        test_dataset.labels.cpu().numpy()
    ])
    class_weights = data_loader.get_class_weights(all_labels)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=CONFIG['batch_size']
    )

    # 2. Create model
    print("\n🧠 Step 2: Creating model...")
    model = create_model(
        input_dim=data_loader.num_features,
        device=CONFIG['device']
    )

    # 3. Train model
    print("\n🎓 Step 3: Training model...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights=class_weights,
        device=CONFIG['device'],
        learning_rate=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    history = trainer.train(epochs=CONFIG['epochs'])

    # 4. Evaluate model
    print("\n📊 Step 4: Evaluating model...")
    metrics = trainer.evaluate()

    # 5. Save final model
    final_model_path = Path('./checkpoints/lstm_gru_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✅ Final model saved to: {final_model_path}")

    # Save metrics
    metrics_path = Path('./checkpoints/test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Test metrics saved to: {metrics_path}")

    print(f"\n🎉 Training pipeline complete!")


if __name__ == "__main__":
    main()
