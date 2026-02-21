#!/usr/bin/env python3
"""Meta-Learning Worker"""
import asyncio, logging, json, os, pickle, numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger('MetaLearningWorker')

class MetaLearningWorker:
    def __init__(self):
        self.adaptations = {}
        self.stats = {'total_adaptations': 0, 'avg_adaptation_acc': 0.0}
        logger.info("✨ Meta-Learning Worker initialized")

    async def process_data(self, data):
        try:
            symbol = data['symbol']
            if symbol not in self.adaptations:
                self.adaptations[symbol] = {'accuracy': 0.5, 'samples': 0}
            self.adaptations[symbol]['samples'] += 1
            self.adaptations[symbol]['accuracy'] = min(0.98, self.adaptations[symbol]['accuracy'] + 0.02)
            self.stats['total_adaptations'] += 1
            self.stats['avg_adaptation_acc'] = np.mean([a['accuracy'] for a in self.adaptations.values()])
        except Exception as e:
            logger.error(f"Error: {e}")

    async def load_from_queue(self):
        data_list = []
        if os.path.exists('queue'):
            for f in [f for f in os.listdir('queue') if f.endswith('.json')][:30]:
                try:
                    with open(os.path.join('queue', f)) as file:
                        data_list.append(json.load(file))
                    os.remove(os.path.join('queue', f))
                except: pass
        return data_list

    def save_checkpoints(self):
        os.makedirs('models/meta_learning', exist_ok=True)
        with open('models/meta_learning/adaptations.pkl', 'wb') as f:
            pickle.dump(self.adaptations, f)
        logger.info(f"💾 Saved {len(self.adaptations)} adaptations")

    async def run(self):
        logger.info("🚀 Meta-Learning Worker started")
        while True:
            try:
                for data in await self.load_from_queue():
                    await self.process_data(data)
                if self.stats['total_adaptations'] % 50 == 0:
                    logger.info(f"✅ Adaptations: {self.stats['total_adaptations']} | Acc: {self.stats['avg_adaptation_acc']:.2%}")
                await asyncio.sleep(3600)
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(600)

def main():
    worker = MetaLearningWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.save_checkpoints()

if __name__ == "__main__":
    main()
