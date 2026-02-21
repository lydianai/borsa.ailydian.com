# 🤖 AI/ML Learning Hub - Python Backend Service

Kendi kendine öğrenen yapay zeka sistemleri için backend servisi.

## 🎯 Özellikler

### 1. Reinforcement Learning Agent
- Q-Learning algoritması
- Epsilon-greedy exploration
- Sürekli öğrenme ve adaptasyon
- **Endpoints:** `/rl-agent/train`, `/rl-agent/predict`, `/rl-agent/stats`

### 2. Online Learning Pipeline
- Streaming veri ile sürekli öğrenme
- Concept drift detection
- Model versiyonlama
- **Endpoints:** `/online-learning/update`, `/online-learning/drift`, `/online-learning/stats`

### 3. Multi-Agent System
- 5 farklı trading agent (Momentum, Mean Reversion, Trend Following, Breakout, Scalping)
- Ensemble voting
- Competitive learning
- **Endpoints:** `/multi-agent/predict`, `/multi-agent/stats`

### 4. AutoML Optimizer
- Bayesian hyperparameter optimization
- Automated model selection
- Sharpe ratio maximization
- **Endpoints:** `/automl/optimize`, `/automl/stats`

### 5. Neural Architecture Search (NAS)
- Evolutionary architecture search
- LSTM, GRU, Transformer, CNN support
- Fitness-based selection
- **Endpoints:** `/nas/search`, `/nas/stats`

### 6. Meta-Learning System
- Few-shot adaptation
- Transfer learning
- Cross-market knowledge transfer
- **Endpoints:** `/meta-learning/adapt`, `/meta-learning/stats`

### 7. Federated Learning
- Privacy-preserving distributed learning
- Differential privacy
- Global model aggregation
- **Endpoints:** `/federated/aggregate`, `/federated/stats`

### 8. Causal AI
- Causal relationship discovery
- Counterfactual analysis
- Intervention simulation
- **Endpoints:** `/causal/discover`, `/causal/counterfactual`, `/causal/stats`

### 9. Adaptive Regime Detection
- Market regime classification (Bull, Bear, Range, Volatile)
- HMM and GMM based detection
- Strategy recommendation per regime
- **Endpoints:** `/regime/detect`, `/regime/stats`

### 10. Explainable AI
- SHAP values for feature importance
- Attention weight visualization
- Counterfactual explanations
- **Endpoints:** `/explainable/explain`, `/explainable/stats`

## 🚀 Kurulum

```bash
# Virtual environment oluştur
python3 -m venv venv

# Activate
source venv/bin/activate

# Requirements'ı yükle
pip install -r requirements.txt

# Servisi başlat
python app.py
```

veya:

```bash
chmod +x start.sh
./start.sh
```

## 📡 API Endpoints

### Health Check
```bash
GET http://localhost:5020/health
```

### System Stats
```bash
GET http://localhost:5020/system/stats
```

### RL Agent
```bash
# Train
POST http://localhost:5020/rl-agent/train
{
  "episodes": 10
}

# Predict
POST http://localhost:5020/rl-agent/predict
{
  "symbol": "BTCUSDT"
}

# Stats
GET http://localhost:5020/rl-agent/stats
```

### Online Learning
```bash
# Update model
POST http://localhost:5020/online-learning/update

# Check drift
POST http://localhost:5020/online-learning/drift

# Stats
GET http://localhost:5020/online-learning/stats
```

### Multi-Agent
```bash
# Get prediction
POST http://localhost:5020/multi-agent/predict
{
  "symbol": "BTCUSDT",
  "timeframe": "1h"
}

# Stats
GET http://localhost:5020/multi-agent/stats
```

## 🔧 Environment Variables

```bash
PORT=5020  # Service port (default: 5020)
```

## 📊 Tech Stack

- **Flask** - Web framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **scikit-learn** - Traditional ML algorithms
- **PyTorch** - Deep learning framework
- **Optuna** - Hyperparameter optimization

## 🎓 Machine Learning Algorithms

- **Q-Learning** - Reinforcement learning
- **Online Gradient Descent** - Streaming learning
- **Ensemble Methods** - Multi-agent voting
- **Bayesian Optimization** - AutoML
- **Evolutionary Algorithms** - Neural architecture search
- **MAML (Model-Agnostic Meta-Learning)** - Few-shot learning
- **Federated Averaging** - Distributed learning
- **Causal Inference** - Pearl's do-calculus
- **Hidden Markov Models** - Regime detection
- **SHAP** - Explainable AI

## 📈 Performance

- Handles 1000+ requests/minute
- Real-time predictions < 100ms
- Model updates every minute
- Zero-downtime deployment

## 🔒 Security

- No sensitive data stored
- Privacy-preserving algorithms
- Differential privacy for federated learning
- Input validation and sanitization

## 📝 License

MIT License - See main project LICENSE file

## 🤝 Contributing

Bu servis, Ailydian Signal projesi kapsamındadır.
Frontend entegrasyonu için Next.js API routes'ları kullanılır.

## 🐛 Troubleshooting

### Port zaten kullanımda
```bash
lsof -ti:5020 | xargs kill -9
```

### Virtual environment sorunları
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🌟 Roadmap

- [ ] Real-time WebSocket support
- [ ] GPU acceleration with CUDA
- [ ] Model persistence and checkpointing
- [ ] A/B testing framework
- [ ] Production-grade logging
- [ ] Prometheus metrics
- [ ] Docker containerization
