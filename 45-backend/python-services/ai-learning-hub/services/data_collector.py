#!/usr/bin/env python3
"""
Data Collector Service
Tüm Binance Futures USDT-M coinleri için sürekli veri toplar
"""

import asyncio
import aiohttp
import logging
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger('DataCollector')

class BinanceFuturesDataCollector:
    """
    Binance Futures market data collector
    """

    def __init__(self):
        self.binance_api = "https://fapi.binance.com/fapi/v1"
        self.symbols: List[str] = []
        self.collection_interval = int(os.getenv('COLLECTION_INTERVAL', '60'))  # 1 dakika
        self.symbols_per_batch = int(os.getenv('SYMBOLS_PER_BATCH', '50'))

        # İstatistikler
        self.stats = {
            'total_collected': 0,
            'successful': 0,
            'failed': 0,
            'start_time': datetime.now()
        }

        logger.info("📡 Data Collector initialized")

    async def fetch_all_symbols(self) -> List[str]:
        """
        Tüm USDT-M perpetual symbollerini çek
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

                    logger.info(f"📊 Loaded {len(symbols)} USDT-M perpetual symbols")
                    return sorted(symbols)
        except Exception as e:
            logger.error(f"❌ Error fetching symbols: {e}")
            return []

    async def fetch_symbol_data(self, session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
        """
        Bir symbol için market data çek
        """
        try:
            # 24h ticker
            async with session.get(
                f"{self.binance_api}/ticker/24hr?symbol={symbol}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    return None

                ticker = await resp.json()

            # Recent klines (son 100 mum - 5m interval)
            async with session.get(
                f"{self.binance_api}/klines?symbol={symbol}&interval=5m&limit=100",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status != 200:
                    klines = []
                else:
                    klines = await resp.json()

            # Funding rate
            async with session.get(
                f"{self.binance_api}/premiumIndex?symbol={symbol}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    funding_data = await resp.json()
                    funding_rate = float(funding_data.get('lastFundingRate', 0))
                else:
                    funding_rate = 0

            # Open interest
            async with session.get(
                f"{self.binance_api}/openInterest?symbol={symbol}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    oi_data = await resp.json()
                    open_interest = float(oi_data.get('openInterest', 0))
                else:
                    open_interest = 0

            return {
                'symbol': symbol,
                'price': float(ticker['lastPrice']),
                'change_24h': float(ticker['priceChangePercent']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'high_24h': float(ticker['highPrice']),
                'low_24h': float(ticker['lowPrice']),
                'trades': int(ticker['count']),
                'funding_rate': funding_rate,
                'open_interest': open_interest,
                'klines': klines,
                'timestamp': datetime.now().isoformat(),
                'collected_at': time.time()
            }

        except Exception as e:
            logger.debug(f"Error fetching {symbol}: {e}")
            return None

    async def collect_batch(self, symbols: List[str]) -> List[Dict]:
        """
        Bir grup sembol için paralel veri toplama
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_symbol_data(session, symbol) for symbol in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = []
            for result in results:
                if result and not isinstance(result, Exception):
                    valid_results.append(result)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1

            return valid_results

    def save_to_queue(self, data_list: List[Dict]):
        """
        Toplanan verileri queue'ya kaydet
        """
        os.makedirs('queue', exist_ok=True)

        for data in data_list:
            symbol = data['symbol']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"queue/{symbol}_{timestamp}.json"

            with open(filename, 'w') as f:
                json.dump(data, f)

        self.stats['total_collected'] += len(data_list)

    async def run(self):
        """
        Ana collection loop
        """
        logger.info("🚀 Data Collector started")
        logger.info(f"⏱️ Collection interval: {self.collection_interval}s")
        logger.info(f"📦 Batch size: {self.symbols_per_batch} symbols")

        # Symboller'i yükle
        self.symbols = await self.fetch_all_symbols()
        if not self.symbols:
            logger.error("❌ No symbols loaded, exiting")
            return

        iteration = 0

        while True:
            try:
                iteration += 1
                cycle_start = time.time()

                logger.info(f"🔄 Iteration #{iteration} - Collecting {len(self.symbols)} symbols...")

                # Batch'lere böl ve topla
                all_data = []
                for i in range(0, len(self.symbols), self.symbols_per_batch):
                    batch = self.symbols[i:i+self.symbols_per_batch]
                    batch_data = await self.collect_batch(batch)
                    all_data.extend(batch_data)

                    logger.info(
                        f"  ✓ Batch {i//self.symbols_per_batch + 1}: "
                        f"{len(batch_data)}/{len(batch)} collected"
                    )

                # Queue'ya kaydet
                if all_data:
                    self.save_to_queue(all_data)

                cycle_time = time.time() - cycle_start
                uptime = (datetime.now() - self.stats['start_time']).total_seconds() / 3600

                logger.info(
                    f"✅ Iteration #{iteration} completed in {cycle_time:.1f}s | "
                    f"Collected: {len(all_data)}/{len(self.symbols)} | "
                    f"Total: {self.stats['total_collected']} | "
                    f"Success Rate: {self.stats['successful']/(self.stats['successful']+self.stats['failed'])*100:.1f}% | "
                    f"Uptime: {uptime:.1f}h"
                )

                # Sonraki cycle için bekle
                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"❌ Error in collection loop: {e}")
                await asyncio.sleep(60)

def main():
    collector = BinanceFuturesDataCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        logger.info("🛑 Data Collector stopped")

if __name__ == "__main__":
    main()
