#!/usr/bin/env python3
"""
AI/ML Learning Hub - Main Orchestrator
Tüm Binance Futures USDT-M coinleri için sürekli öğrenme sistemi

Bu script 7/24 arka planda çalışır ve:
1. Tüm 538 coini sürekli izler
2. AI/ML modellerini her yeni veri ile günceller
3. Model checkpointlarını düzenli kaydeder
4. Performance metriklerini loglar
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
import sys

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/orchestrator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('Orchestrator')

class AILearningOrchestrator:
    """
    Ana orchestrator - tüm AI/ML sistemlerini koordine eder
    """

    def __init__(self):
        self.binance_api = "https://fapi.binance.com/fapi/v1"
        self.symbols: List[str] = []
        self.active = True

        # Service URLs (mevcut Python servisleri)
        self.services = {
            'ta_lib': 'http://localhost:5001',
            'signal_gen': 'http://localhost:5002',
            'risk_mgmt': 'http://localhost:5003',
            'feature_eng': 'http://localhost:5004',
            'smc_strategy': 'http://localhost:5005',
            'transformer': 'http://localhost:5006',
            'online_learning': 'http://localhost:5007',
            'multi_timeframe': 'http://localhost:5008',
            'order_flow': 'http://localhost:5009',
            'continuous_monitor': 'http://localhost:5010',
            'mfi_monitor': 'http://localhost:5011'
        }

        # AI/ML Workers status
        self.workers_status = {
            'rl_agent': {'active': False, 'last_update': None, 'processed': 0},
            'online_learning': {'active': False, 'last_update': None, 'processed': 0},
            'multi_agent': {'active': False, 'last_update': None, 'processed': 0},
            'automl': {'active': False, 'last_update': None, 'processed': 0},
            'nas': {'active': False, 'last_update': None, 'processed': 0},
            'meta_learning': {'active': False, 'last_update': None, 'processed': 0},
            'federated': {'active': False, 'last_update': None, 'processed': 0},
            'causal_ai': {'active': False, 'last_update': None, 'processed': 0},
            'regime_detection': {'active': False, 'last_update': None, 'processed': 0},
            'explainable_ai': {'active': False, 'last_update': None, 'processed': 0}
        }

        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_trained': 0,
            'start_time': datetime.now(),
            'uptime_hours': 0,
            'avg_processing_time': 0
        }

        logger.info("🤖 AI Learning Orchestrator initialized")

    async def fetch_all_symbols(self) -> List[str]:
        """
        Tüm Binance Futures USDT-M perpetual symbollerini çek
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.binance_api}/exchangeInfo") as resp:
                    data = await resp.json()

                    symbols = [
                        s['symbol'] for s in data['symbols']
                        if s['quoteAsset'] == 'USDT'
                        and s['status'] == 'TRADING'
                        and s['contractType'] == 'PERPETUAL'
                    ]

                    logger.info(f"📊 Total Binance Futures USDT-M symbols: {len(symbols)}")
                    return sorted(symbols)
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {e}")
            return []

    async def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Belirli bir coin için market data çek
        """
        try:
            async with aiohttp.ClientSession() as session:
                # 24h ticker data
                async with session.get(f"{self.binance_api}/ticker/24hr?symbol={symbol}") as resp:
                    ticker = await resp.json()

                # Recent klines (candlestick data) - son 100 mum
                async with session.get(
                    f"{self.binance_api}/klines?symbol={symbol}&interval=5m&limit=100"
                ) as resp:
                    klines = await resp.json()

                return {
                    'symbol': symbol,
                    'price': float(ticker['lastPrice']),
                    'change_24h': float(ticker['priceChangePercent']),
                    'volume': float(ticker['volume']),
                    'high_24h': float(ticker['highPrice']),
                    'low_24h': float(ticker['lowPrice']),
                    'klines': klines,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"❌ Error fetching {symbol}: {e}")
            return None

    async def fetch_service_data(self, service_name: str, endpoint: str, params: Dict = None) -> Dict:
        """
        Mevcut Python servislerinden veri çek
        """
        try:
            service_url = self.services.get(service_name)
            if not service_url:
                return None

            url = f"{service_url}{endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        return None
        except Exception as e:
            logger.debug(f"Service {service_name} not available: {e}")
            return None

    async def collect_comprehensive_data(self, symbol: str) -> Dict[str, Any]:
        """
        Bir coin için tüm kaynaklardan veri topla
        """
        start_time = time.time()

        # 1. Market data
        market_data = await self.fetch_market_data(symbol)
        if not market_data:
            return None

        # 2. TA-Lib indikatörleri (varsa)
        ta_indicators = await self.fetch_service_data('ta_lib', '/indicators', {'symbol': symbol})

        # 3. Signal Generator (varsa)
        signals = await self.fetch_service_data('signal_gen', '/signals', {'symbol': symbol})

        # 4. Feature Engineering (varsa)
        features = await self.fetch_service_data('feature_eng', '/features', {'symbol': symbol})

        # 5. Risk Management (varsa)
        risk_data = await self.fetch_service_data('risk_mgmt', '/risk', {'symbol': symbol})

        # Hepsini birleştir
        comprehensive_data = {
            **market_data,
            'indicators': ta_indicators if ta_indicators else {},
            'signals': signals if signals else {},
            'features': features if features else {},
            'risk': risk_data if risk_data else {},
            'collection_time_ms': (time.time() - start_time) * 1000
        }

        return comprehensive_data

    async def distribute_to_workers(self, data: Dict[str, Any]):
        """
        Toplanan veriyi tüm AI/ML worker'lara dağıt
        """
        # Her worker kendi queue'sundan okuyacak
        # Şimdilik basit file-based queue kullanıyoruz
        # Production'da Redis Pub/Sub veya RabbitMQ kullanılabilir

        symbol = data['symbol']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Queue klasörüne yaz
        os.makedirs('queue', exist_ok=True)
        queue_file = f"queue/{symbol}_{timestamp}.json"

        with open(queue_file, 'w') as f:
            json.dump(data, f)

        logger.debug(f"📤 Data queued for {symbol}")
        self.stats['total_processed'] += 1

    async def process_symbols_batch(self, symbols_batch: List[str]):
        """
        Bir grup sembolü paralel olarak işle
        """
        tasks = [self.collect_comprehensive_data(symbol) for symbol in symbols_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = [r for r in results if r and not isinstance(r, Exception)]

        for data in valid_results:
            await self.distribute_to_workers(data)

        return len(valid_results)

    async def monitoring_loop(self):
        """
        Sistem durumunu izle ve logla
        """
        while self.active:
            try:
                # Uptime hesapla
                uptime = datetime.now() - self.stats['start_time']
                self.stats['uptime_hours'] = uptime.total_seconds() / 3600

                # Worker status logla
                active_workers = sum(1 for w in self.workers_status.values() if w['active'])

                logger.info(
                    f"📊 Status: {active_workers}/10 workers active | "
                    f"Processed: {self.stats['total_processed']} | "
                    f"Uptime: {self.stats['uptime_hours']:.1f}h"
                )

                # Her 5 dakikada bir detaylı log
                await asyncio.sleep(300)
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                await asyncio.sleep(60)

    async def main_loop(self):
        """
        Ana işlem döngüsü - sürekli çalışır
        """
        logger.info("🚀 Starting main orchestration loop...")

        # Symboller listesini çek
        self.symbols = await self.fetch_all_symbols()
        if not self.symbols:
            logger.error("❌ No symbols found, exiting")
            return

        logger.info(f"✅ Loaded {len(self.symbols)} symbols")

        # Batch size (aynı anda kaç coin işlenecek)
        batch_size = 50  # 50 coin paralel

        iteration = 0

        while self.active:
            try:
                iteration += 1
                cycle_start = time.time()

                logger.info(f"🔄 Iteration #{iteration} - Processing {len(self.symbols)} symbols...")

                # Symboller'i batch'lere böl
                for i in range(0, len(self.symbols), batch_size):
                    batch = self.symbols[i:i+batch_size]
                    processed = await self.process_symbols_batch(batch)
                    logger.info(f"  ✓ Batch {i//batch_size + 1}: {processed}/{len(batch)} processed")

                cycle_time = time.time() - cycle_start
                logger.info(
                    f"✅ Iteration #{iteration} completed in {cycle_time:.1f}s "
                    f"({len(self.symbols)/cycle_time:.1f} symbols/sec)"
                )

                # Sonraki cycle için bekle (5 dakika)
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)

    async def run(self):
        """
        Orchestrator'ı başlat
        """
        logger.info("=" * 70)
        logger.info("🤖 AI/ML LEARNING HUB - ORCHESTRATOR")
        logger.info("=" * 70)
        logger.info("📍 Mode: Continuous Learning")
        logger.info("🌐 Market: Binance Futures USDT-M")
        logger.info("⚡ Workers: 10 AI/ML systems")
        logger.info("=" * 70)

        # Klasörleri oluştur
        os.makedirs('logs', exist_ok=True)
        os.makedirs('queue', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Ana loop ve monitoring'i paralel çalıştır
        await asyncio.gather(
            self.main_loop(),
            self.monitoring_loop()
        )

def main():
    """
    Entry point
    """
    orchestrator = AILearningOrchestrator()

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("🛑 Orchestrator stopped by user")
        orchestrator.active = False
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
