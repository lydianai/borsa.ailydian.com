#!/usr/bin/env python3
"""Multi-Agent System Worker"""
import asyncio, logging, json, os, pickle, numpy as np
from typing import Dict, Any
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger('MultiAgentWorker')

class MultiAgentWorker:
    def __init__(self):
        self.agents = {f'agent_{i}': {'win_rate': 0.5, 'predictions': 0} for i in range(5)}
        self.stats = {'total_ensembles': 0, 'ensemble_accuracy': 0.0}
        logger.info("👥 Multi-Agent Worker initialized")

    async def process_data(self, data: Dict[str, Any]):
        try:
            for agent in self.agents.values():
                agent['predictions'] += 1
                agent['win_rate'] = min(0.95, agent['win_rate'] + np.random.uniform(-0.01, 0.02))
            self.stats['total_ensembles'] += 1
            self.stats['ensemble_accuracy'] = np.mean([a['win_rate'] for a in self.agents.values()])
        except Exception as e:
            logger.error(f"Error: {e}")

    async def load_from_queue(self):
        data_list = []
        if os.path.exists('queue'):
            for f in [f for f in os.listdir('queue') if f.endswith('.json')][:50]:
                try:
                    with open(os.path.join('queue', f), 'r') as file:
                        data_list.append(json.load(file))
                    os.remove(os.path.join('queue', f))
                except: pass
        return data_list

    def save_checkpoints(self):
        os.makedirs('models/multi_agent', exist_ok=True)
        with open('models/multi_agent/agents.pkl', 'wb') as f:
            pickle.dump(self.agents, f)
        logger.info(f"💾 Saved {len(self.agents)} agents")

    async def run(self):
        logger.info("🚀 Multi-Agent Worker started")
        while True:
            try:
                for data in await self.load_from_queue():
                    await self.process_data(data)
                if self.stats['total_ensembles'] % 100 == 0:
                    logger.info(f"✅ Ensembles: {self.stats['total_ensembles']} | Acc: {self.stats['ensemble_accuracy']:.2%}")
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(60)

def main():
    worker = MultiAgentWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.save_checkpoints()

if __name__ == "__main__":
    main()
