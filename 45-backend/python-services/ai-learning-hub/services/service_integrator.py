#!/usr/bin/env python3
"""
Service Integrator
Tüm Python servislerini entegre eder ve veri toplar
"""

import asyncio
import aiohttp
import logging
import json
import os
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
logger = logging.getLogger('ServiceIntegrator')

class ServiceIntegrator:
    """Tüm mevcut Python servislerini entegre eder"""

    def __init__(self):
        self.services = {
            'ta_lib': {'url': 'http://localhost:5001', 'available': False},
            'signal_gen': {'url': 'http://localhost:5002', 'available': False},
            'risk_mgmt': {'url': 'http://localhost:5003', 'available': False},
            'feature_eng': {'url': 'http://localhost:5004', 'available': False},
            'smc_strategy': {'url': 'http://localhost:5005', 'available': False},
            'transformer': {'url': 'http://localhost:5006', 'available': False},
            'online_learning': {'url': 'http://localhost:5007', 'available': False},
            'multi_timeframe': {'url': 'http://localhost:5008', 'available': False},
            'order_flow': {'url': 'http://localhost:5009', 'available': False},
            'continuous_monitor': {'url': 'http://localhost:5010', 'available': False},
            'mfi_monitor': {'url': 'http://localhost:5011', 'available': False}
        }
        self.stats = {'total_requests': 0, 'successful': 0, 'failed': 0}
        logger.info("🔗 Service Integrator initialized")

    async def check_service_health(self, service_name: str, url: str) -> bool:
        """Servis sağlık kontrolü"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    return resp.status == 200
        except:
            return False

    async def discover_services(self):
        """Tüm servislerin durumunu kontrol et"""
        logger.info("🔍 Discovering services...")
        tasks = [self.check_service_health(name, info['url']) for name, info in self.services.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (name, info), available in zip(self.services.items(), results):
            info['available'] = available if not isinstance(available, Exception) else False
            status = "✅" if info['available'] else "❌"
            logger.info(f"  {status} {name:20s} → {info['url']}")

        available_count = sum(1 for info in self.services.values() if info['available'])
        logger.info(f"📊 Services available: {available_count}/{len(self.services)}")

    async def fetch_from_service(self, service_name: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Belirli bir servisten veri çek"""
        service = self.services.get(service_name)
        if not service or not service['available']:
            return None

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{service['url']}{endpoint}"
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        self.stats['successful'] += 1
                        return await resp.json()
                    else:
                        self.stats['failed'] += 1
                        return None
        except Exception as e:
            self.stats['failed'] += 1
            logger.debug(f"Error fetching from {service_name}: {e}")
            return None

    async def enrich_data(self, base_data: Dict[str, Any]) -> Dict[str, Any]:
        """Market datasını tüm servislerden gelen bilgilerle zenginleştir"""
        symbol = base_data.get('symbol')
        if not symbol:
            return base_data

        self.stats['total_requests'] += 1

        # Paralel olarak tüm servislerden veri topla
        tasks = {
            'ta_indicators': self.fetch_from_service('ta_lib', '/indicators', {'symbol': symbol}),
            'signals': self.fetch_from_service('signal_gen', '/signals', {'symbol': symbol}),
            'risk': self.fetch_from_service('risk_mgmt', '/risk', {'symbol': symbol}),
            'features': self.fetch_from_service('feature_eng', '/features', {'symbol': symbol}),
            'smc': self.fetch_from_service('smc_strategy', '/orderblocks', {'symbol': symbol}),
            'transformer': self.fetch_from_service('transformer', '/attention', {'symbol': symbol}),
            'drift': self.fetch_from_service('online_learning', '/drift', {'symbol': symbol}),
            'timeframes': self.fetch_from_service('multi_timeframe', '/analysis', {'symbol': symbol}),
            'order_flow': self.fetch_from_service('order_flow', '/profile', {'symbol': symbol}),
            'mfi': self.fetch_from_service('mfi_monitor', '/mfi', {'symbol': symbol})
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        enriched = {**base_data}

        for key, result in zip(tasks.keys(), results):
            if result and not isinstance(result, Exception):
                enriched[key] = result

        return enriched

    async def run(self):
        """Ana servis döngüsü"""
        logger.info("🚀 Service Integrator started")

        # İlk service discovery
        await self.discover_services()

        # Her 5 dakikada bir service health check
        last_discovery = asyncio.get_event_loop().time()

        while True:
            try:
                current_time = asyncio.get_event_loop().time()

                # 5 dakikada bir service discovery
                if current_time - last_discovery >= 300:
                    await self.discover_services()
                    last_discovery = current_time

                # İstatistikleri logla
                if self.stats['total_requests'] % 100 == 0 and self.stats['total_requests'] > 0:
                    success_rate = self.stats['successful'] / (self.stats['successful'] + self.stats['failed']) * 100
                    logger.info(
                        f"📊 Requests: {self.stats['total_requests']} | "
                        f"Success Rate: {success_rate:.1f}%"
                    )

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)

def main():
    integrator = ServiceIntegrator()
    try:
        asyncio.run(integrator.run())
    except KeyboardInterrupt:
        logger.info("🛑 Service Integrator stopped")

if __name__ == "__main__":
    main()
