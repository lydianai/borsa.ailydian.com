#!/usr/bin/env python3
"""
Reinforcement Learning Agent Worker
Sürekli öğrenen RL ajanı - tüm coinler için tahmin ve öğrenme yapar
"""

import asyncio
import logging
import json
import os
import time
import pickle
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('RLWorker')

class RLAgentWorker:
    """
    RL Agent sürekli öğrenme worker'ı
    """

    def __init__(self):
        self.q_tables = {}  # Her symbol için ayrı Q-table
        self.training_interval = int(os.getenv('TRAINING_INTERVAL', '300'))  # 5 dakika
        self.checkpoint_interval = int(os.getenv('CHECKPOINT_INTERVAL', '3600'))  # 1 saat
        self.last_checkpoint = time.time()

        # RL parametreleri
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        self.epsilon = 0.1  # Exploration rate

        # İstatistikler
        self.stats = {
            'total_episodes': 0,
            'total_rewards': 0,
            'win_rate': 0.0,
            'processed_symbols': set()
        }

        logger.info("⚡ RL Agent Worker initialized")

    def get_or_create_q_table(self, symbol: str) -> Dict:
        """
        Symbol için Q-table oluştur veya yükle
        """
        if symbol not in self.q_tables:
            # Checkpoint'ten yükle
            checkpoint_path = f"models/rl_agent/{symbol}_q_table.pkl"
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'rb') as f:
                    self.q_tables[symbol] = pickle.load(f)
                logger.info(f"📥 Loaded Q-table for {symbol}")
            else:
                # Yeni Q-table oluştur
                self.q_tables[symbol] = {}
                logger.info(f"🆕 Created new Q-table for {symbol}")

        return self.q_tables[symbol]

    def discretize_state(self, data: Dict[str, Any]) -> str:
        """
        Continuous state'i discrete state'e çevir
        """
        try:
            price = data['price']
            change_24h = data['change_24h']
            volume = data['volume']

            # RSI (varsa)
            rsi = data.get('indicators', {}).get('rsi', 50)

            # State'i kategorize et
            trend = 'up' if change_24h > 1 else 'down' if change_24h < -1 else 'neutral'
            rsi_level = 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            vol_level = 'high' if volume > 1000000 else 'low'

            state = f"{trend}_{rsi_level}_{vol_level}"
            return state
        except Exception as e:
            logger.error(f"Error discretizing state: {e}")
            return "unknown"

    def choose_action(self, q_table: Dict, state: str) -> str:
        """
        Epsilon-greedy action selection
        """
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(['BUY', 'SELL', 'HOLD'])
        else:
            # Exploitation
            if state not in q_table:
                q_table[state] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            return max(q_table[state], key=q_table[state].get)

    def calculate_reward(self, action: str, data: Dict, next_data: Dict = None) -> float:
        """
        Reward hesapla (basitleştirilmiş)
        """
        if next_data is None:
            # Simulated reward (gerçek implementation'da next_data beklemeli)
            change = data['change_24h']
            if action == 'BUY' and change > 0:
                return change / 10  # Pozitif reward
            elif action == 'SELL' and change < 0:
                return abs(change) / 10  # Pozitif reward
            elif action == 'HOLD':
                return 0.01  # Small positive for holding
            else:
                return -abs(change) / 10  # Negative reward
        else:
            # Gerçek reward hesaplama
            price_change = (next_data['price'] - data['price']) / data['price'] * 100
            if action == 'BUY':
                return price_change
            elif action == 'SELL':
                return -price_change
            else:
                return 0

    def update_q_table(self, q_table: Dict, state: str, action: str, reward: float, next_state: str):
        """
        Q-Learning güncelleme
        """
        if state not in q_table:
            q_table[state] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        if next_state not in q_table:
            q_table[next_state] = {'BUY': 0, 'SELL': 0, 'HOLD': 0}

        # Q-Learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        old_value = q_table[state][action]
        next_max = max(q_table[next_state].values())
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        q_table[state][action] = new_value

    async def process_data(self, data: Dict[str, Any]):
        """
        Bir data point'i işle
        """
        try:
            symbol = data['symbol']
            self.stats['processed_symbols'].add(symbol)

            # Q-table al
            q_table = self.get_or_create_q_table(symbol)

            # State'i hesapla
            state = self.discretize_state(data)

            # Action seç
            action = self.choose_action(q_table, state)

            # Reward hesapla (simulated)
            reward = self.calculate_reward(action, data)

            # Q-table güncelle (next_state şu an aynı state, gerçekte bir sonraki data olmalı)
            next_state = state  # Simplified
            self.update_q_table(q_table, state, action, reward, next_state)

            # İstatistikleri güncelle
            self.stats['total_episodes'] += 1
            self.stats['total_rewards'] += reward

            if reward > 0:
                wins = self.stats.get('wins', 0) + 1
                self.stats['wins'] = wins
                self.stats['win_rate'] = wins / self.stats['total_episodes'] * 100

            logger.debug(
                f"🎯 {symbol}: state={state}, action={action}, "
                f"reward={reward:.2f}, Q-size={len(q_table)}"
            )

        except Exception as e:
            logger.error(f"❌ Error processing {data.get('symbol', 'unknown')}: {e}")

    async def load_from_queue(self) -> List[Dict]:
        """
        Queue'dan veri yükle
        """
        queue_files = []
        if os.path.exists('queue'):
            queue_files = [f for f in os.listdir('queue') if f.endswith('.json')]

        data_list = []
        for filename in queue_files[:100]:  # Max 100 file at a time
            filepath = os.path.join('queue', filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    data_list.append(data)
                # İşlendikten sonra sil
                os.remove(filepath)
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

        return data_list

    def save_checkpoints(self):
        """
        Tüm Q-table'ları kaydet
        """
        os.makedirs('models/rl_agent', exist_ok=True)

        for symbol, q_table in self.q_tables.items():
            checkpoint_path = f"models/rl_agent/{symbol}_q_table.pkl"
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(q_table, f)

        # İstatistikleri kaydet
        stats_path = "models/rl_agent/stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                **self.stats,
                'processed_symbols': list(self.stats['processed_symbols'])
            }, f, indent=2)

        logger.info(
            f"💾 Checkpoint saved: {len(self.q_tables)} Q-tables, "
            f"{self.stats['total_episodes']} episodes"
        )

    async def run(self):
        """
        Worker ana loop
        """
        logger.info("🚀 RL Agent Worker started")
        logger.info(f"⚙️ Training interval: {self.training_interval}s")
        logger.info(f"💾 Checkpoint interval: {self.checkpoint_interval}s")

        while True:
            try:
                # Queue'dan veri yükle
                data_list = await self.load_from_queue()

                if data_list:
                    logger.info(f"📊 Processing {len(data_list)} data points...")

                    # Her veriyi işle
                    for data in data_list:
                        await self.process_data(data)

                    logger.info(
                        f"✅ Processed {len(data_list)} | "
                        f"Episodes: {self.stats['total_episodes']} | "
                        f"Win Rate: {self.stats['win_rate']:.1f}% | "
                        f"Symbols: {len(self.stats['processed_symbols'])}"
                    )

                # Checkpoint gerekli mi?
                if time.time() - self.last_checkpoint >= self.checkpoint_interval:
                    self.save_checkpoints()
                    self.last_checkpoint = time.time()

                # Bekle
                await asyncio.sleep(self.training_interval)

            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)

def main():
    worker = RLAgentWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("🛑 RL Agent Worker stopped")
        worker.save_checkpoints()

if __name__ == "__main__":
    main()
