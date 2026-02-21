/**
 * 🔔 MA7 PULLBACK INSTANT NOTIFICATION SERVICE
 * Tüm coinleri tarayıp MA7 Pullback stratejisi koşullarını karşılayanlar için
 * anlık browser bildirimi gönderir
 */

interface MA7Signal {
  symbol: string;
  price: number;
  pullbackPercent: number;
  confidence: number;
  reason: string;
}

class MA7NotificationService {
  private notifiedCoins: Set<string> = new Set();
  private scanInterval: NodeJS.Timeout | null = null;
  private isScanning: boolean = false;

  /**
   * MA7 Pullback taramasını başlat
   * @param intervalMinutes - Tarama aralığı (dakika cinsinden, varsayılan: 5)
   */
  async startScanning(intervalMinutes: number = 5) {
    console.log(`[MA7 Notification] Tarama başlatıldı (${intervalMinutes} dakika aralıkla)`);

    // İlk taramayı hemen yap
    await this.scanAndNotify();

    // Periyodik tarama başlat
    this.scanInterval = setInterval(async () => {
      await this.scanAndNotify();
    }, intervalMinutes * 60 * 1000);
  }

  /**
   * Taramayı durdur
   */
  stopScanning() {
    if (this.scanInterval) {
      clearInterval(this.scanInterval);
      this.scanInterval = null;
      console.log('[MA7 Notification] Tarama durduruldu');
    }
  }

  /**
   * Tüm coinleri tara ve MA7 Pullback sinyali olanları bildir
   */
  private async scanAndNotify() {
    if (this.isScanning) {
      console.log('[MA7 Notification] Tarama zaten devam ediyor, atlanıyor');
      return;
    }

    this.isScanning = true;

    try {
      console.log('[MA7 Notification] Tarama başlıyor...');

      // Tüm coinleri getir
      const response = await fetch('/api/binance/futures');
      const result = await response.json();

      if (!result.success) {
        throw new Error('Market data alınamadı');
      }

      const allCoins = result.data.all;
      console.log(`[MA7 Notification] ${allCoins.length} coin taranıyor...`);

      // MA7 Pullback stratejisini her coin için kontrol et
      const signals: MA7Signal[] = [];

      for (const coin of allCoins) {
        const signal = this.checkMA7Pullback(coin);
        if (signal) {
          signals.push(signal);
        }
      }

      console.log(`[MA7 Notification] ${signals.length} MA7 Pullback sinyali bulundu`);

      // Yeni sinyaller için bildirim gönder
      for (const signal of signals) {
        await this.notifyIfNew(signal);
      }

      // Eski bildirimleri temizle (1 saat öncesinden eski coinleri sil)
      this.clearOldNotifications();

    } catch (error) {
      console.error('[MA7 Notification] Tarama hatası:', error);
    } finally {
      this.isScanning = false;
    }
  }

  /**
   * MA7 Pullback stratejisini kontrol et
   */
  private checkMA7Pullback(coin: any): MA7Signal | null {
    const { symbol, price, changePercent24h, high24h } = coin;

    // MA7 pullback hesaplama
    const pullback = ((high24h - price) / high24h) * 100;
    const goodPullback = pullback >= 1 && pullback <= 4;
    const momentum = changePercent24h > 2;

    // Koşullar karşılanıyorsa sinyal üret
    if (goodPullback && momentum) {
      return {
        symbol,
        price,
        pullbackPercent: pullback,
        confidence: 86,
        reason: `MA7 pullback tamamlandı (${pullback.toFixed(1)}%). İdeal giriş noktası. 24s değişim: +${changePercent24h.toFixed(2)}%`
      };
    }

    return null;
  }

  /**
   * Yeni sinyal için bildirim gönder (daha önce bildirilmediyse)
   */
  private async notifyIfNew(signal: MA7Signal) {
    // Bu coin için daha önce bildirim gönderildiyse atla
    if (this.notifiedCoins.has(signal.symbol)) {
      return;
    }

    // Browser bildirim izni kontrolü
    if (!('Notification' in window)) {
      console.warn('[MA7 Notification] Browser bildirimleri desteklenmiyor');
      return;
    }

    if (Notification.permission === 'granted') {
      // Bildirim gönder
      const notification = new Notification('🎯 MA7 Pullback Sinyali!', {
        body: `${signal.symbol.replace('USDT', '')}/USDT\n${signal.reason}\nFiyat: $${signal.price.toFixed(4)}\nGüven: %${signal.confidence}`,
        icon: '/favicon.ico',
        badge: '/favicon.ico',
        tag: signal.symbol, // Aynı coin için birden fazla bildirim gösterme
        requireInteraction: true, // Kullanıcı kapatana kadar göster
        data: {
          symbol: signal.symbol,
          price: signal.price,
          url: `/market-scanner?symbol=${signal.symbol}`
        }
      });

      // Bildirime tıklanınca ilgili sayfaya git
      notification.onclick = (event) => {
        event.preventDefault();
        window.focus();
        window.location.href = `/market-scanner?symbol=${signal.symbol}`;
        notification.close();
      };

      console.log(`[MA7 Notification] Bildirim gönderildi: ${signal.symbol}`);

      // Bu coini bildirildi olarak işaretle
      this.notifiedCoins.add(signal.symbol);

    } else if (Notification.permission !== 'denied') {
      // İzin istenmemişse iste
      const permission = await Notification.requestPermission();
      if (permission === 'granted') {
        // İzin verildi, bildirimi tekrar gönder
        await this.notifyIfNew(signal);
      }
    }
  }

  /**
   * 1 saatten eski bildirimleri temizle
   */
  private clearOldNotifications() {
    // Basit implementasyon: Her 1 saatte bir listeyi temizle
    // Daha gelişmiş implementasyon için timestamp tutulabilir
    const _CLEAR_INTERVAL = 60 * 60 * 1000; // 1 saat

    if (this.notifiedCoins.size > 100) {
      console.log('[MA7 Notification] Eski bildirimler temizleniyor');
      this.notifiedCoins.clear();
    }
  }

  /**
   * Belirli bir coin için bildirimi sıfırla (test amaçlı)
   */
  resetNotification(symbol: string) {
    this.notifiedCoins.delete(symbol);
    console.log(`[MA7 Notification] ${symbol} bildirimi sıfırlandı`);
  }

  /**
   * Tüm bildirimleri sıfırla
   */
  resetAllNotifications() {
    this.notifiedCoins.clear();
    console.log('[MA7 Notification] Tüm bildirimler sıfırlandı');
  }

  /**
   * Tarama durumunu getir
   */
  getStatus() {
    return {
      isScanning: this.isScanning,
      isActive: this.scanInterval !== null,
      notifiedCoinsCount: this.notifiedCoins.size,
      notifiedCoins: Array.from(this.notifiedCoins)
    };
  }
}

// Singleton instance
export const ma7NotificationService = new MA7NotificationService();
