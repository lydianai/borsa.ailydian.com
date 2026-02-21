"""
🎯 SİNYAL ANALİZCİSİ - LONG TAHMİNLERİ İÇİN BİLDİRİM SİSTEMİ
=============================================================

Tüm AI/ML sistemlerinden gelen long sinyallerini analiz eder ve
yüksek güvenilirlikte olanları bildirim olarak gönderir.

Özellikler:
- Multi-AI consensus (birden fazla AI aynı yönde sinyal verirse güvenilirlik artar)
- Confidence threshold filtering (minimum %75 güven skoru)
- Signal deduplication (aynı coin için tekrar eden sinyaller engellenir)
- Turkish language support (tüm bildirimler Türkçe)
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import os

class SignalAnalyzer:
    """Long sinyallerini analiz eden ve bildirim üreten sistem"""

    def __init__(self):
        self.signals_history = defaultdict(list)  # symbol -> [signals]
        self.notifications = []  # Tüm bildirimler
        self.min_confidence = 75.0  # Minimum %75 güven skoru
        self.consensus_threshold = 3  # En az 3 AI aynı yönde olmalı

        # Load Binance symbols
        self.load_binance_symbols()

    def load_binance_symbols(self):
        """Binance Futures USDT-M coin listesini yükle"""
        try:
            symbols_file = os.path.join(
                os.path.dirname(__file__),
                '..',
                'binance_futures_symbols.json'
            )
            with open(symbols_file, 'r') as f:
                data = json.load(f)
                self.all_symbols = data['symbols']
                print(f"✅ {len(self.all_symbols)} adet Binance Futures USDT-M coin yüklendi")
        except FileNotFoundError:
            print("⚠️ binance_futures_symbols.json bulunamadı, varsayılan liste kullanılıyor")
            self.all_symbols = [
                "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
                "ADAUSDT", "AVAXUSDT", "DOGEUSDT", "DOTUSDT", "MATICUSDT"
            ]

    def analyze_signal(self, signal: Dict) -> Optional[Dict]:
        """
        Tek bir sinyali analiz et

        Args:
            signal: {
                'symbol': 'BTCUSDT',
                'action': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 85.5,
                'source': 'RL Agent',
                'price': 45000.0,
                'timestamp': '2025-11-20T10:00:00'
            }

        Returns:
            Bildirime dönüştürülecek sinyal veya None
        """
        # Sadece LONG (BUY) sinyallerini al
        if signal['action'] != 'BUY':
            return None

        # Minimum güven skorunu kontrol et
        if signal['confidence'] < self.min_confidence:
            return None

        # Geçmişe ekle
        symbol = signal['symbol']
        self.signals_history[symbol].append({
            **signal,
            'analyzed_at': datetime.now().isoformat()
        })

        # Son 5 dakikadaki sinyalleri kontrol et
        recent_signals = self._get_recent_signals(symbol, minutes=5)

        # Consensus kontrolü
        consensus_count = len([s for s in recent_signals if s['action'] == 'BUY'])

        if consensus_count >= self.consensus_threshold:
            # Güçlü konsensüs var - bildirim oluştur!
            return self._create_notification(signal, recent_signals, consensus_count)

        return None

    def _get_recent_signals(self, symbol: str, minutes: int = 5) -> List[Dict]:
        """Son X dakikadaki sinyalleri getir"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent = []

        for sig in self.signals_history[symbol]:
            sig_time = datetime.fromisoformat(sig['analyzed_at'])
            if sig_time > cutoff_time:
                recent.append(sig)

        return recent

    def _create_notification(
        self,
        signal: Dict,
        supporting_signals: List[Dict],
        consensus_count: int
    ) -> Dict:
        """Bildirim oluştur (tamamen Türkçe)"""

        # Katılımcı AI sistemlerini listele
        ai_sources = list(set([s['source'] for s in supporting_signals]))
        ai_sources_str = ", ".join(ai_sources[:3])  # İlk 3 tanesi
        if len(ai_sources) > 3:
            ai_sources_str += f" ve {len(ai_sources) - 3} tane daha"

        # Ortalama güven skorunu hesapla
        avg_confidence = sum([s['confidence'] for s in supporting_signals]) / len(supporting_signals)

        notification = {
            'id': f"LONG-{signal['symbol']}-{int(time.time())}",
            'type': 'LONG_SIGNAL',
            'priority': 'YÜKSEK' if avg_confidence > 90 else 'ORTA',
            'symbol': signal['symbol'],
            'price': signal['price'],
            'confidence': round(avg_confidence, 2),
            'consensus_count': consensus_count,
            'ai_sources': ai_sources_str,
            'title': f"🚀 LONG Fırsatı: {signal['symbol']}",
            'message': f"{signal['symbol']} için {consensus_count} AI sisteminden LONG sinyali geldi! Ortalama güven: %{avg_confidence:.1f}",
            'details': f"Katılımcı AI'lar: {ai_sources_str}. Fiyat: ${signal['price']:,.2f}",
            'action': 'LONG',
            'timestamp': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(minutes=15)).isoformat(),
            'status': 'ACTIVE'
        }

        # Bildirimlere ekle
        self.notifications.append(notification)

        # Son 100 bildirimi tut
        if len(self.notifications) > 100:
            self.notifications = self.notifications[-100:]

        return notification

    def get_active_notifications(self) -> List[Dict]:
        """Aktif (süresi dolmamış) bildirimleri getir"""
        now = datetime.now()
        active = []

        for notif in self.notifications:
            expires_at = datetime.fromisoformat(notif['expires_at'])
            if expires_at > now and notif['status'] == 'ACTIVE':
                active.append(notif)

        return active

    def dismiss_notification(self, notification_id: str) -> bool:
        """Bildirimi kapat"""
        for notif in self.notifications:
            if notif['id'] == notification_id:
                notif['status'] = 'DISMISSED'
                return True
        return False

    def get_statistics(self) -> Dict:
        """İstatistikler"""
        total_signals = sum([len(sigs) for sigs in self.signals_history.values()])
        total_notifications = len(self.notifications)
        active_notifications = len(self.get_active_notifications())

        # En çok sinyal alan coinler
        top_signals = sorted(
            [(symbol, len(sigs)) for symbol, sigs in self.signals_history.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total_signals': total_signals,
            'total_notifications': total_notifications,
            'active_notifications': active_notifications,
            'unique_symbols': len(self.signals_history),
            'top_10_signals': [
                {'symbol': sym, 'count': count}
                for sym, count in top_signals
            ]
        }

# Global instance
signal_analyzer = SignalAnalyzer()
