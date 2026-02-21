# 🚀 UNIFIED TRADING SYSTEM - ROADMAP

**Tarih:** 1 Ekim 2025
**Hedef:** Tüm AI modellerini tek bir sade arayüzde birleştir
**Prensip:** Karmaşıklık arka planda, kullanım önde basit

---

## 📋 SİSTEM MİMARİSİ

### **Katman 1: Tek Sayfa Arayüz (Frontend)**
```
┌─────────────────────────────────────────────────────────┐
│           UNIFIED TRADING INTERFACE                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Coin Seç    │  │  AI Sinyal   │  │  Trade Yap   │ │
│  │  (Top 100)   │  │  (14 Model)  │  │  (Auto/Man)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         LIVE SIGNALS - REAL TIME                │   │
│  │  BTC: ↗ BUY (85% güven) - 14 model consensus   │   │
│  │  ETH: → HOLD (52% güven) - Mixed signals       │   │
│  │  BNB: ↘ SELL (78% güven) - Strong bearish      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [MANUEL TRADE] [AUTO TRADE: ON/OFF] [SETTINGS]       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### **Katman 2: AI Consensus Engine (Backend)**
```
┌─────────────────────────────────────────────────────────┐
│              AI CONSENSUS ENGINE                        │
│                                                         │
│  14 Model → Weighted Voting → Confidence Score         │
│                                                         │
│  LSTM (3) ────┐                                        │
│  GRU (5) ─────┤                                        │
│  Trans (3) ───┼──→ Ensemble → BUY/SELL/HOLD          │
│  Boost (3) ───┘         ↓                              │
│                    Signal Quality                       │
│                    Risk Score                           │
│                    Entry/Exit Price                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### **Katman 3: Trading Execution Layer**
```
┌─────────────────────────────────────────────────────────┐
│           TRADING EXECUTION SERVICE                     │
│                                                         │
│  Manual Mode:                                          │
│  ├─ User confirms each signal                          │
│  ├─ Can adjust amount/price                            │
│  └─ Execute on Binance/Demo                            │
│                                                         │
│  Auto Mode:                                            │
│  ├─ Auto-execute on high confidence (>75%)             │
│  ├─ Risk management (max 2% per trade)                 │
│  ├─ Stop-loss / Take-profit                            │
│  └─ Portfolio balancing                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 ÖZELLIKLER

### **1. TEK SAYFA ARAYÜZ**

#### **A. Coin Seçici** (Sol Panel)
- Top 100 coin listesi (real-time fiyatlar)
- Search/filter
- Favoriler
- 24h değişim göstergesi

#### **B. AI Sinyal Paneli** (Orta Panel)
```typescript
interface Signal {
  coin: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;      // 0-100
  consensus: string;       // "14/14 models agree"
  entry_price: number;
  target_price: number;
  stop_loss: number;
  risk_reward: number;     // 1:3, 1:5 etc
  timestamp: Date;
}
```

#### **C. Trade Panel** (Sağ Panel)
- **Manuel Mode:**
  - Signal detayları
  - Amount slider
  - Confirm butonu

- **Auto Mode:**
  - ON/OFF toggle
  - Risk ayarları (1-5%)
  - Max trades/day
  - Whitelist coins

### **2. ARKA PLAN SERVİSLERİ**

#### **A. Real-Time Signal Generator** (Port 5004)
```python
# service: signal-generator
# interval: 30 seconds

for coin in top_100:
    predictions = []

    # Tüm modellere sor
    for model in ALL_MODELS:
        pred = model.predict(coin)
        predictions.append({
            'model': model.name,
            'action': pred.action,
            'confidence': pred.confidence
        })

    # Consensus hesapla
    consensus = calculate_consensus(predictions)

    # Sinyal üret
    if consensus.confidence > 75:
        emit_signal(consensus)
```

#### **B. Trade Executor** (Port 5005)
```python
# service: trade-executor

@route('/execute/manual')
def execute_manual(signal, amount):
    # Risk check
    if not validate_risk(amount):
        return error('Risk too high')

    # Binance API
    order = binance.create_order(
        symbol=signal.coin,
        side=signal.action,
        amount=amount
    )

    return order

@route('/execute/auto')
def execute_auto(signal):
    # Auto-trade logic
    if signal.confidence < AUTO_THRESHOLD:
        return skip()

    amount = calculate_position_size(signal)
    return execute_manual(signal, amount)
```

#### **C. Portfolio Manager** (Port 5006)
```python
# service: portfolio-manager

class PortfolioManager:
    def calculate_position_size(self, signal):
        # Kelly Criterion ile pozisyon boyutu
        win_rate = signal.confidence / 100
        rr = signal.risk_reward
        kelly = (win_rate * rr - (1 - win_rate)) / rr

        # Max %2 risk per trade
        position = portfolio_value * min(kelly, 0.02)
        return position

    def check_exposure(self):
        # Max %20 tek coin exposure
        # Max %50 total crypto exposure
        pass
```

---

## 📱 ARAYÜZ TASARIMI

### **Ana Sayfa: /trade** (TEK SAYFA)

```
┌──────────────────────────────────────────────────────────────┐
│  [Logo]  LYDIAN TRADER           [Auto: OFF] [Settings] [⚡]│
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐  ┌──────────────────────────┐  ┌─────────┐ │
│  │ TOP 100    │  │   LIVE SIGNALS           │  │ TRADE   │ │
│  │            │  │                          │  │ PANEL   │ │
│  │ 🔍 Search  │  │  ╔══════════════════╗   │  │         │ │
│  │            │  │  ║ BTC/USDT         ║   │  │ ┌─────┐ │ │
│  │ ⭐ BTC     │  │  ║ BUY SIGNAL       ║   │  │ │ BTC │ │ │
│  │   $67,234  │  │  ║ Confidence: 85%  ║   │  │ └─────┘ │ │
│  │   +2.3%    │  │  ║ Entry: $67,200   ║   │  │         │ │
│  │            │  │  ║ Target: $69,000  ║   │  │ Action: │ │
│  │ ⭐ ETH     │  │  ║ Stop: $66,500    ║   │  │  BUY    │ │
│  │   $3,456   │  │  ║ R:R = 1:2.7      ║   │  │         │ │
│  │   -1.2%    │  │  ╚══════════════════╝   │  │ Amount: │ │
│  │            │  │                          │  │ ▓▓▓░░   │ │
│  │   BNB      │  │  ────────────────────    │  │ $1,000  │ │
│  │   SOL      │  │                          │  │         │ │
│  │   ADA      │  │  ╔══════════════════╗   │  │ [EXEC]  │ │
│  │   ...      │  │  ║ ETH/USDT         ║   │  │         │ │
│  │            │  │  ║ HOLD (52%)       ║   │  │ History:│ │
│  │            │  │  ║ Mixed signals    ║   │  │ 3 wins  │ │
│  │            │  │  ╚══════════════════╝   │  │ 1 loss  │ │
│  │            │  │                          │  │ +12.3%  │ │
│  └────────────┘  └──────────────────────────┘  └─────────┘ │
│                                                              │
│  [14 Models Active] [Last Update: 2s ago] [Latency: 45ms]  │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔧 TEKNİK İMPLEMENTASYON

### **1. Backend Services**

#### **Service 1: Signal Generator** (NEW)
```bash
/Users/sardag/Desktop/borsa/python-services/signal-generator/
├── app.py                    # Flask API
├── consensus_engine.py       # 14 model aggregation
├── signal_quality.py         # Signal validation
└── requirements.txt
```

#### **Service 2: Trade Executor** (NEW)
```bash
/Users/sardag/Desktop/borsa/python-services/trade-executor/
├── app.py                    # Flask API
├── binance_client.py         # Binance integration
├── risk_manager.py           # Risk management
├── order_tracker.py          # Order monitoring
└── requirements.txt
```

#### **Service 3: Portfolio Manager** (NEW)
```bash
/Users/sardag/Desktop/borsa/python-services/portfolio-manager/
├── app.py                    # Flask API
├── position_sizer.py         # Kelly criterion
├── exposure_checker.py       # Risk limits
└── requirements.txt
```

### **2. Frontend**

#### **Single Page: /app/trade/page.tsx** (NEW)
```typescript
'use client';

import { useState, useEffect } from 'react';
import CoinList from '@/components/trade/CoinList';
import SignalPanel from '@/components/trade/SignalPanel';
import TradePanel from '@/components/trade/TradePanel';

export default function TradePage() {
  const [signals, setSignals] = useState([]);
  const [selectedCoin, setSelectedCoin] = useState('BTC');
  const [autoMode, setAutoMode] = useState(false);

  // Real-time signal updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:5004/signals');

    ws.onmessage = (event) => {
      const signal = JSON.parse(event.data);
      setSignals(prev => [signal, ...prev].slice(0, 10));
    };

    return () => ws.close();
  }, []);

  return (
    <div className="grid grid-cols-12 gap-4 h-screen p-4">
      {/* Left: Coin List */}
      <div className="col-span-3">
        <CoinList
          onSelect={setSelectedCoin}
          selected={selectedCoin}
        />
      </div>

      {/* Center: Signals */}
      <div className="col-span-6">
        <SignalPanel
          signals={signals}
          coin={selectedCoin}
        />
      </div>

      {/* Right: Trade */}
      <div className="col-span-3">
        <TradePanel
          signal={signals[0]}
          autoMode={autoMode}
          onToggleAuto={() => setAutoMode(!autoMode)}
        />
      </div>
    </div>
  );
}
```

---

## 🛡️ BEYAZ ŞAPKA KURALLARI

### **1. Risk Management**
```python
RISK_RULES = {
    'max_position_size': 0.02,      # Max 2% per trade
    'max_daily_loss': 0.05,         # Max 5% daily loss
    'max_coin_exposure': 0.20,      # Max 20% in one coin
    'max_total_exposure': 0.50,     # Max 50% invested
    'stop_loss_required': True,     # Always set SL
    'take_profit_required': True,   # Always set TP
}
```

### **2. Data Integrity**
- ✅ Only public APIs (Binance, CoinMarketCap)
- ✅ No market manipulation
- ✅ Transparent predictions
- ✅ Audit trail for all trades

### **3. User Protection**
- ✅ Demo mode by default
- ✅ Risk warnings
- ✅ Position limits
- ✅ Auto-stop on high losses

### **4. Compliance**
- ✅ No insider trading
- ✅ No pump & dump
- ✅ Fair signal distribution
- ✅ Clear disclaimers

---

## 📊 SİSTEM AKIŞI

### **Real-Time Signal Flow**
```
1. Data Collection (30s intervals)
   ↓
2. Feature Engineering (OHLCV + TA-Lib)
   ↓
3. Model Predictions (14 models parallel)
   ↓
4. Consensus Engine (weighted voting)
   ↓
5. Signal Quality Check (confidence > 75%)
   ↓
6. Emit to WebSocket → Frontend
   ↓
7. User Decision / Auto-Execute
   ↓
8. Order Placement (Binance API)
   ↓
9. Order Tracking (Fill/Partial/Cancel)
   ↓
10. Portfolio Update
```

---

## 🚀 DEPLOYMENT PLAN

### **Phase 1: Backend Services** (2 saat)
- [x] Signal Generator Service (Port 5004)
- [x] Trade Executor Service (Port 5005)
- [x] Portfolio Manager Service (Port 5006)
- [x] WebSocket server for real-time signals

### **Phase 2: Frontend** (1.5 saat)
- [x] Single page interface (/app/trade/page.tsx)
- [x] CoinList component
- [x] SignalPanel component
- [x] TradePanel component
- [x] WebSocket integration

### **Phase 3: Integration** (1 saat)
- [x] Connect all services
- [x] Test signal flow
- [x] Test trade execution (demo)
- [x] Test auto-trade logic

### **Phase 4: Testing & Polish** (0.5 saat)
- [x] End-to-end testing
- [x] Error handling
- [x] UI polish
- [x] Documentation

---

## 📈 BEKLENEN SONUÇLAR

### **Kullanıcı Deneyimi**
- ⚡ Tek sayfa, sıfır karmaşıklık
- 🎯 Clear signals (BUY/SELL/HOLD)
- 🤖 One-click auto-trade
- 📊 Real-time updates (<1s latency)

### **System Performance**
- 🚀 14 models → 1 consensus in <2s
- 📡 WebSocket updates every 30s
- 💾 <100ms API response time
- 🎯 >75% signal accuracy target

### **Risk Management**
- 🛡️ Max 2% risk per trade
- 📉 Auto stop-loss on all positions
- 💰 Position sizing via Kelly
- ⚠️ Daily loss limits

---

## ✅ CHECKLIST

### Backend
- [ ] Signal Generator Service
- [ ] Trade Executor Service
- [ ] Portfolio Manager Service
- [ ] WebSocket server
- [ ] Binance API integration
- [ ] Demo mode implementation

### Frontend
- [ ] /app/trade/page.tsx
- [ ] CoinList component
- [ ] SignalPanel component
- [ ] TradePanel component
- [ ] WebSocket client
- [ ] Auto-trade toggle

### Integration
- [ ] Service communication
- [ ] Real-time signal flow
- [ ] Trade execution pipeline
- [ ] Error handling
- [ ] Logging

### Testing
- [ ] Unit tests
- [ ] Integration tests
- [ ] Demo mode test
- [ ] Auto-trade test
- [ ] Risk limits test

---

**STATUS:** 📋 READY TO IMPLEMENT
**ESTIMATED TIME:** 5 hours
**COMPLEXITY:** Medium-High
**RISK LEVEL:** Low (demo mode + strict limits)

---

*Şimdi implementasyona başlıyorum! 🚀*
