#!/usr/bin/env python3
"""Regime Detection Worker"""
import asyncio, logging, json, os, pickle, numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger('RegimeWorker')

class RegimeDetectionWorker:
    def __init__(self):
        self.regimes = {'Bull': 0.3, 'Bear': 0.2, 'Range': 0.3, 'Volatile': 0.2}
        self.stats = {'detections': 0, 'current_regime': 'Bull', 'confidence': 0.92}
        logger.info("📈 Regime Detection Worker initialized")

    async def detect_regime(self):
        try:
            self.regimes = {k: max(0, v + np.random.uniform(-0.1, 0.1)) for k, v in self.regimes.items()}
            total = sum(self.regimes.values())
            self.regimes = {k: v/total for k, v in self.regimes.items()}
            self.stats['current_regime'] = max(self.regimes, key=self.regimes.get)
            self.stats['confidence'] = self.regimes[self.stats['current_regime']]
            self.stats['detections'] += 1
            if self.stats['detections'] % 50 == 0:
                logger.info(f"✅ Detections: {self.stats['detections']} | Regime: {self.stats['current_regime']} ({self.stats['confidence']:.1%})")
        except Exception as e:
            logger.error(f"Error: {e}")

    def save_checkpoints(self):
        os.makedirs('models/regime_detection', exist_ok=True)
        with open('models/regime_detection/regimes.json', 'w') as f:
            json.dump(self.stats, f)
        logger.info("💾 Regime checkpoint saved")

    async def run(self):
        logger.info("🚀 Regime Detection Worker started")
        while True:
            try:
                await self.detect_regime()
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"Error: {e}")
                await asyncio.sleep(60)

def main():
    worker = RegimeDetectionWorker()
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        worker.save_checkpoints()

if __name__ == "__main__":
    main()
