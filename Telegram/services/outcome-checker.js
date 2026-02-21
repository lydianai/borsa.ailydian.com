/**
 * 🔍 OUTCOME CHECKER SERVICE
 *
 * Her saat başı çalışarak 24 saat geçmiş sinyallerin sonuçlarını kontrol eder:
 * - Binance 24h price history'den high/low fiyatları çeker
 * - TP1/TP2/TP3 veya Stop Loss'a ulaşıldı mı kontrol eder
 * - Signal sonuçlarını (SUCCESS/FAILURE) günceller
 * - PnL hesaplar
 *
 * PM2 Cron: Her saat başı çalıştır
 * pm2 start outcome-checker.js --cron "0 * * * *"
 */

const https = require('https');
const { getPendingSignals, updateSignalOutcome } = require('./signal-tracker');

// ===== CONFIGURATION =====
const BINANCE_API_BASE = 'api.binance.com';

// ===== BINANCE API: KLINES (HISTORICAL CANDLES) =====
/**
 * Binance'den son 24 saatlik kline (mum) verisini çek
 * @param {string} symbol - Coin sembolü (örn: BTCUSDT)
 * @param {string} startTime - Başlangıç zamanı (timestamp ms)
 * @returns {Promise<Object>} - { highPrice, lowPrice, closePrice }
 */
async function getBinance24hPriceHistory(symbol, startTime) {
  return new Promise((resolve, reject) => {
    const endTime = Date.now();
    const interval = '1h'; // 1 saatlik mumlar (24 mum = 24 saat)

    const path = `/fapi/v1/klines?symbol=${symbol}&interval=${interval}&startTime=${startTime}&endTime=${endTime}&limit=24`;

    const options = {
      hostname: BINANCE_API_BASE,
      port: 443,
      path: path,
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 10000,
    };

    const req = https.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          if (res.statusCode === 200) {
            const klines = JSON.parse(data);

            if (!klines || klines.length === 0) {
              reject(new Error('No kline data received'));
              return;
            }

            // Kline format: [openTime, open, high, low, close, volume, closeTime, ...]
            // index 2 = high, index 3 = low, index 4 = close

            let highPrice = -Infinity;
            let lowPrice = Infinity;
            let closePrice = 0;

            klines.forEach(kline => {
              const high = parseFloat(kline[2]);
              const low = parseFloat(kline[3]);
              const close = parseFloat(kline[4]);

              if (high > highPrice) highPrice = high;
              if (low < lowPrice) lowPrice = low;
              closePrice = close; // Son mum'un close fiyatı
            });

            resolve({ highPrice, lowPrice, closePrice });
          } else {
            reject(new Error(`Binance API error: ${res.statusCode} - ${data}`));
          }
        } catch (error) {
          reject(new Error(`Failed to parse Binance response: ${error.message}`));
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    req.end();
  });
}

// ===== CHECK SIGNAL OUTCOME =====
/**
 * Signal'in sonucunu kontrol et
 * @param {Object} signal - Signal objesi
 * @returns {Promise<Object>} - Outcome bilgileri
 */
async function checkSignalOutcome(signal) {
  try {
    console.log(`🔍 Kontrol ediliyor: ${signal.symbol} (${signal.signalType}) - ${signal.id}`);

    // Binance'den 24h price history çek
    const signalTimestamp = new Date(signal.timestamp).getTime();
    const priceHistory = await getBinance24hPriceHistory(signal.symbol, signalTimestamp);

    const { highPrice, lowPrice, closePrice } = priceHistory;

    console.log(`   📊 24h Fiyat Aralığı: $${lowPrice.toFixed(4)} - $${highPrice.toFixed(4)}`);
    console.log(`   📍 Entry: $${signal.entryPrice.toFixed(4)}`);
    console.log(`   🎯 TP1: $${signal.tp1.toFixed(4)} | TP2: $${signal.tp2.toFixed(4)} | TP3: $${signal.tp3.toFixed(4)}`);
    console.log(`   🛑 Stop Loss: $${signal.stopLoss.toFixed(4)}`);

    let status = 'EXPIRED';
    let outcome = null;
    let pnl = 0;

    // AL (LONG) sinyali için kontrol
    if (signal.signalType === 'AL') {
      // 1. Stop Loss kontrolü (en düşük fiyat SL'a ulaştı mı?)
      if (lowPrice <= signal.stopLoss) {
        status = 'SL_HIT';
        outcome = 'FAILURE';
        pnl = ((signal.stopLoss - signal.entryPrice) / signal.entryPrice) * 100;
        console.log(`   ❌ Stop Loss'a ulaşıldı! PnL: ${pnl.toFixed(2)}%`);
      }
      // 2. TP3 kontrolü
      else if (highPrice >= signal.tp3) {
        status = 'TP3_HIT';
        outcome = 'SUCCESS';
        // Weighted PnL: 40% @ TP1, 30% @ TP2, 30% @ TP3
        const pnlTP1 = ((signal.tp1 - signal.entryPrice) / signal.entryPrice) * 100 * 0.4;
        const pnlTP2 = ((signal.tp2 - signal.entryPrice) / signal.entryPrice) * 100 * 0.3;
        const pnlTP3 = ((signal.tp3 - signal.entryPrice) / signal.entryPrice) * 100 * 0.3;
        pnl = pnlTP1 + pnlTP2 + pnlTP3;
        console.log(`   ✅ TP3'e ulaşıldı! PnL: ${pnl.toFixed(2)}%`);
      }
      // 3. TP2 kontrolü
      else if (highPrice >= signal.tp2) {
        status = 'TP2_HIT';
        outcome = 'SUCCESS';
        const pnlTP1 = ((signal.tp1 - signal.entryPrice) / signal.entryPrice) * 100 * 0.4;
        const pnlTP2 = ((signal.tp2 - signal.entryPrice) / signal.entryPrice) * 100 * 0.3;
        // Kalan %30 entry fiyatından çıkıldı varsayalım
        const pnlRemaining = 0; // Breakeven
        pnl = pnlTP1 + pnlTP2 + pnlRemaining;
        console.log(`   ✅ TP2'ye ulaşıldı! PnL: ${pnl.toFixed(2)}%`);
      }
      // 4. TP1 kontrolü
      else if (highPrice >= signal.tp1) {
        status = 'TP1_HIT';
        outcome = 'SUCCESS';
        const pnlTP1 = ((signal.tp1 - signal.entryPrice) / signal.entryPrice) * 100 * 0.4;
        // Kalan %60 entry fiyatından çıkıldı varsayalım
        const pnlRemaining = 0; // Breakeven
        pnl = pnlTP1 + pnlRemaining;
        console.log(`   ✅ TP1'e ulaşıldı! PnL: ${pnl.toFixed(2)}%`);
      }
      // 5. Hiçbir hedefe ulaşılamadı (EXPIRED)
      else {
        status = 'EXPIRED';
        // Close fiyatı ile entry'yi karşılaştır
        pnl = ((closePrice - signal.entryPrice) / signal.entryPrice) * 100;
        outcome = pnl >= 0 ? 'SUCCESS' : 'FAILURE';
        console.log(`   ⏱️ Süresi doldu. Son fiyat PnL: ${pnl.toFixed(2)}% → ${outcome}`);
      }
    }
    // BEKLE (HOLD) sinyali - pozisyon açılmadı, success sayılır ama PnL 0
    else if (signal.signalType === 'BEKLE') {
      status = 'EXPIRED';
      outcome = 'SUCCESS'; // BEKLE sinyalleri her zaman başarılı (false positive yok)
      pnl = 0;
      console.log(`   ⏸️ BEKLE sinyali - Pozisyon açılmadı (SUCCESS)`);
    }

    return {
      status,
      outcome,
      actualHighPrice: highPrice,
      actualLowPrice: lowPrice,
      pnl: parseFloat(pnl.toFixed(2)),
    };

  } catch (error) {
    console.error(`   ❌ Outcome check hatası: ${error.message}`);
    throw error;
  }
}

// ===== MAIN EXECUTION =====
async function main() {
  const startTime = Date.now();
  console.log('\n╔════════════════════════════════════════════════════════╗');
  console.log('║  🔍 OUTCOME CHECKER - BAŞLADI                         ║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log(`⏰ Başlangıç: ${new Date().toLocaleString('tr-TR')}\n`);

  try {
    // 1. Pending signals'ları getir
    console.log('📡 Kontrol edilecek sinyaller getiriliyor...');
    const pendingSignals = getPendingSignals();

    console.log(`✅ ${pendingSignals.length} adet kontrol edilecek sinyal bulundu.\n`);

    if (pendingSignals.length === 0) {
      console.log('ℹ️ Kontrol edilecek sinyal yok. İşlem sonlandırıldı.\n');
      process.exit(0);
    }

    let successCount = 0;
    let failureCount = 0;
    let errorCount = 0;

    // 2. Her signal için outcome check
    for (const signal of pendingSignals) {
      try {
        const outcomeData = await checkSignalOutcome(signal);

        // Update signal in database
        const updated = updateSignalOutcome(signal.id, outcomeData);

        if (updated) {
          if (outcomeData.outcome === 'SUCCESS') {
            successCount++;
          } else if (outcomeData.outcome === 'FAILURE') {
            failureCount++;
          }
        } else {
          errorCount++;
          console.error(`   ❌ Signal güncelleme başarısız: ${signal.id}`);
        }

        // Rate limiting: Wait 200ms between requests
        await new Promise(resolve => setTimeout(resolve, 200));

      } catch (error) {
        errorCount++;
        console.error(`❌ Signal kontrol hatası: ${signal.symbol} - ${error.message}`);
      }
    }

    const totalTime = Date.now() - startTime;

    console.log('\n╔════════════════════════════════════════════════════════╗');
    console.log('║  ✅ OUTCOME CHECKER - TAMAMLANDI                      ║');
    console.log('╚════════════════════════════════════════════════════════╝');
    console.log(`📊 Özet:`);
    console.log(`   - Kontrol edilen: ${pendingSignals.length} sinyal`);
    console.log(`   - Başarılı: ${successCount} (SUCCESS)`);
    console.log(`   - Başarısız: ${failureCount} (FAILURE)`);
    console.log(`   - Hata: ${errorCount}`);
    console.log(`   - Toplam süre: ${totalTime}ms (${(totalTime / 1000).toFixed(1)}s)`);
    console.log(`⏰ Bitiş: ${new Date().toLocaleString('tr-TR')}\n`);

    // Exit successfully
    process.exit(0);

  } catch (error) {
    console.error('\n╔════════════════════════════════════════════════════════╗');
    console.error('║  ❌ HATA OLUŞTU!                                      ║');
    console.error('╚════════════════════════════════════════════════════════╝');
    console.error(`❌ Hata detayı: ${error.message}`);
    console.error(`📍 Stack: ${error.stack}\n`);

    // Exit with error
    process.exit(1);
  }
}

// ===== RUN =====
if (require.main === module) {
  main();
}

module.exports = { main, checkSignalOutcome, getBinance24hPriceHistory };
