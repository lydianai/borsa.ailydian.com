"""
Model Test Script
=================

Test LSTM-GRU Hybrid model with dummy data
Verify forward pass, prediction, and attention mechanism
"""

import torch
import numpy as np
from lstm_gru_hybrid import create_model, count_parameters

def test_model_creation():
    """Test 1: Model creation"""
    print("="*60)
    print("TEST 1: MODEL CREATION")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = create_model(input_dim=150, device=device)

    params = count_parameters(model)
    print(f"Total parameters: {params:,}")

    assert params == 1139011, f"Expected 1,139,011 parameters, got {params:,}"
    print("✅ Model creation successful\n")

    return model, device


def test_forward_pass(model, device):
    """Test 2: Forward pass"""
    print("="*60)
    print("TEST 2: FORWARD PASS")
    print("="*60)

    batch_size = 8
    seq_len = 60
    input_dim = 150

    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    print(f"Input shape: {x.shape}")

    # Forward pass
    try:
        logits, attention_weights = model(x)
        print(f"Logits shape: {logits.shape}")
        print(f"Attention weights shape: {attention_weights.shape}")

        assert logits.shape == (batch_size, 3), f"Expected logits shape (8, 3), got {logits.shape}"
        assert attention_weights.shape[0] == batch_size, f"Unexpected attention shape"

        print("✅ Forward pass successful\n")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        raise


def test_prediction(model, device):
    """Test 3: Prediction with probabilities"""
    print("="*60)
    print("TEST 3: PREDICTION")
    print("="*60)

    batch_size = 4
    seq_len = 60
    input_dim = 150

    x = torch.randn(batch_size, seq_len, input_dim).to(device)

    try:
        predictions, probabilities, attention_weights = model.predict(x)

        print(f"Predictions shape: {predictions.shape}")
        print(f"Probabilities shape: {probabilities.shape}")

        # Check predictions are valid classes (0, 1, or 2)
        assert torch.all((predictions >= 0) & (predictions <= 2)), "Invalid predictions"

        # Check probabilities sum to 1
        prob_sums = probabilities.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities don't sum to 1"

        print(f"\nSample predictions: {predictions.cpu().numpy()}")
        print(f"Sample probabilities:\n{probabilities.cpu().numpy()}")

        class_names = ['BUY', 'SELL', 'HOLD']
        for i in range(batch_size):
            pred_class = class_names[predictions[i].item()]
            confidence = probabilities[i, predictions[i]].item()
            print(f"  Sample {i+1}: {pred_class} (confidence: {confidence:.2%})")

        print("\n✅ Prediction successful\n")

    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise


def test_attention_mechanism(model, device):
    """Test 4: Attention mechanism"""
    print("="*60)
    print("TEST 4: ATTENTION MECHANISM")
    print("="*60)

    seq_len = 60
    x = torch.randn(1, seq_len, 150).to(device)

    try:
        attention_weights = model.get_attention_weights(x)

        print(f"Attention weights shape: {attention_weights.shape}")

        # Attention weights should be (batch=1, num_heads, seq_len, seq_len)
        assert attention_weights.shape[0] == 1, "Unexpected batch size"
        assert attention_weights.shape[2] == seq_len, "Unexpected sequence length"
        assert attention_weights.shape[3] == seq_len, "Unexpected attention dimension"

        # Average attention across heads for last timestep
        attention_avg = attention_weights.mean(dim=1)[0, -1, :].cpu().numpy()

        print(f"\nAttention distribution (last timestep):")
        print(f"  Sum: {attention_avg.sum():.4f} (should be ~1.0)")
        print(f"  Min: {attention_avg.min():.6f}")
        print(f"  Max: {attention_avg.max():.6f}")
        print(f"  Mean: {attention_avg.mean():.6f}")

        # Find top 5 most attended timesteps
        top_indices = np.argsort(attention_avg)[-5:][::-1]
        print(f"\nTop 5 most attended timesteps:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i}. Timestep {idx}: {attention_avg[idx]:.4f}")

        print("\n✅ Attention mechanism successful\n")

    except Exception as e:
        print(f"❌ Attention mechanism failed: {e}")
        raise


def test_batch_sizes(model, device):
    """Test 5: Different batch sizes"""
    print("="*60)
    print("TEST 5: DIFFERENT BATCH SIZES")
    print("="*60)

    batch_sizes = [1, 2, 4, 8, 16, 32]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 60, 150).to(device)

        try:
            logits, _ = model(x)
            assert logits.shape == (batch_size, 3), f"Failed for batch size {batch_size}"
            print(f"✅ Batch size {batch_size:2d}: OK")
        except Exception as e:
            print(f"❌ Batch size {batch_size:2d}: FAILED - {e}")
            raise

    print("\n✅ All batch sizes successful\n")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 LSTM-GRU HYBRID MODEL TESTS")
    print("="*60 + "\n")

    try:
        # Test 1: Model creation
        model, device = test_model_creation()

        # Test 2: Forward pass
        test_forward_pass(model, device)

        # Test 3: Prediction
        test_prediction(model, device)

        # Test 4: Attention mechanism
        test_attention_mechanism(model, device)

        # Test 5: Different batch sizes
        test_batch_sizes(model, device)

        # Summary
        print("="*60)
        print("🎉 ALL TESTS PASSED")
        print("="*60)
        print("\n✅ Model is ready for training!")
        print(f"✅ Device: {device}")
        print(f"✅ Parameters: 1,139,011")
        print(f"✅ Input: (batch, 60, 150)")
        print(f"✅ Output: (batch, 3) - BUY/SELL/HOLD")
        print(f"✅ Attention: Multi-head (4 heads)")
        print("\n")

    except Exception as e:
        print("\n" + "="*60)
        print("❌ TESTS FAILED")
        print("="*60)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
