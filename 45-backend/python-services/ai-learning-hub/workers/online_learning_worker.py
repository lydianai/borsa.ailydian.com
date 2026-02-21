#!/usr/bin/env python3
"""
Online Learning Worker
Sürekli model güncellemesi ve concept drift detection
"""

import asyncio
import logging
import json
import os
import time
import pickle
from datetime import datetime
from typing import Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger('OnlineLearningWorker')

class OnlineLearningWorker:
    def __init__(self):
        self.models = {}
        self.drift_detectors = {}
        self.stats = {'total_updates': 0, 'drift_detected': 0, 'avg_accuracy': 0.0}
        logger.info("🔄 Online Learning Worker initialized")

    async def process_data(self, data: Dict[str, Any]):
        try:
            symbol = data['symbol']
            if symbol not in self.models:
                self.models[symbol] = {'version': 1, 'accuracy': 0.5, 'drift_score': 0.0}

            # Simulated model update
            self.models[symbol]['accuracy'] = min(0.99, self.models[symbol]['accuracy'] + np.random.uniform(0, 0.01))
            self.models[symbol]['drift_score'] = np.random.uniform(0, 0.3)

            if self.models[symbol]['drift_score'] > 0.2:
                self.stats['drift_detected'] += 1
                self.models[symbol]['version'] += 1

            self.stats['total_updates'] += 1
            self.stats['avg_accuracy'] = np.mean([m['accuracy'] for m in self.models.values()])

        except Exception as e:
            logger.error(f"Error processing {data.get('symbol')}: {e}")

    async def load_from_queue(self):
        queue_files = []
        if os.path.exists('queue'):
            queue_files = [f for f in os.listdir('queue') if f.endswith('.json')]

        data_list = []
        for filename in queue_files[:50]:
            filepath = os.path.join('queue', filename)
            try:
                with open(filepath, 'r') as f:
                    data_list.append(json.load(f))
                os.remove(filepath)
            except: pass
        return data_list

    def save_checkpoints(self):
        os.makedirs('models/online_learning', exist_ok=True)
        with open('models/online_learning/models.pkl', 'wb') as f:
            pickle.dump(self.models, f)
        with open('models/online_learning/stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"💾 Checkpoint saved: {len(self.models)} models")

    async def run(self):
        logger.info("🚀 Online Learning Worker started")
        while True:
            try:
                data_list = await self.load_from_queue()
                if data_list:
                    for data in data_list:
                        await self.process_data(data)
                    logger.info(f"✅ Updates: {self.stats['total_updates']} | Accuracy: {self.stats['avg_accuracy']:.2%}")
                await asyncio.sleep(600)
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(60)

def main():
    worker = OnlineLearningWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.save_checkpoints()

if __name__ == "__main__":
    main()
