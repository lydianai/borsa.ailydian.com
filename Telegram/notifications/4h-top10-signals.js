/**
 * 📊 4 SAATLİK TOP 10 SİNYAL BİLDİRİM SERVİSİ
 *
 * Trader'lar için her 4 saatte bir TOP 10 coin'leri analiz ederek Telegram'a gönderir:
 * - Haftalık değişim + hacim bazlı TOP 10 seçimi
 * - AL/BEKLE sinyalleri
 * - Giriş/Çıkış/TP/SL fiyatları
 * - Kaldıraç ve sermaye önerileri
 * - Risk yönetimi detayları
 *
 * PM2 ile çalıştırılır: pm2 start 4h-top10-signals.js --cron "5 */4 * * *"
 * Cron: Her 4 saatte bir, 5. dakikada (00:05, 04:05, 08:05, 12:05, 16:05, 20:05)
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');
const { trackSignal } = require('../services/signal-tracker');

// ===== CONFIGURATION =====
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || '8292640150:AAHqDdkHxFqx9q8hJ-bJ8KS_Z2LZWrOLroI';
const SUBSCRIBERS_FILE = path.join(__dirname, '../', 'subscribers.json');
const CONFIG_FILE = path.join(__dirname, '../', 'config.json');
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';

// ===== ADMIN-ONLY MODE CHECK =====
function getConfig() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      return JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
    }
  } catch (error) {
    console.error(`⚠️ Config dosyası okunamadı: ${error.message}`);
  }

  return {
    adminChatId: 7575640489,
    adminOnly: false,
    performanceTracking: true
  };
}

// ===== SUBSCRIBER MANAGEMENT =====
function getActiveSubscribers() {
  const config = getConfig();

  // Admin-only mode: Sadece admin'e bildirim gönder
  if (config.adminOnly) {
    console.log(`⚠️ ADMIN-ONLY MODE ACTIVE: Sadece admin (${config.adminChatId}) bildirim alacak`);
    return [config.adminChatId.toString()];
  }

  // Normal mode: Tüm aktif aboneler
  try {
    if (fs.existsSync(SUBSCRIBERS_FILE)) {
      const data = JSON.parse(fs.readFileSync(SUBSCRIBERS_FILE, 'utf8'));
      const activeSubscribers = data.subscribers
        .filter(sub => sub.active)
        .map(sub => sub.chatId.toString());

      console.log(`✅ ${activeSubscribers.length} aktif abone bulundu`);
      return activeSubscribers;
    }
  } catch (error) {
    console.error(`⚠️ Subscribers dosyası okunamadı: ${error.message}`);
  }

  const envChatIds = (process.env.TELEGRAM_ALLOWED_CHAT_IDS || '7575640489').split(',');
  console.log(`⚠️ Fallback: ${envChatIds.length} abone env'den alındı`);
  return envChatIds;
}

const TELEGRAM_CHAT_IDS = getActiveSubscribers();

// ===== TELEGRAM API =====
async function sendTelegramMessage(chatId, message, options = {}) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      chat_id: chatId,
      text: message,
      parse_mode: 'Markdown',
      disable_web_page_preview: true,
      ...options,
    });

    const req = https.request(
      {
        hostname: 'api.telegram.org',
        port: 443,
        path: `/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
      },
      (res) => {
        let responseData = '';
        res.on('data', (chunk) => {
          responseData += chunk;
        });
        res.on('end', () => {
          if (res.statusCode === 200) {
            resolve(JSON.parse(responseData));
          } else {
            reject(new Error(`Telegram API error: ${res.statusCode} - ${responseData}`));
          }
        });
      }
    );

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

// ===== FETCH TOP 10 SIGNALS =====
async function fetchTop10Signals() {
  return new Promise((resolve, reject) => {
    const url = `${API_BASE_URL}/api/telegram/top10-4h-signals`;
    const parsedUrl = new URL(url);

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 90000, // 90 seconds (complex analysis takes time)
    };

    const client = parsedUrl.protocol === 'https:' ? https : http;

    const req = client.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const jsonData = JSON.parse(data);
          if (jsonData.success) {
            resolve(jsonData.data);
          } else {
            reject(new Error(jsonData.error || 'API returned success: false'));
          }
        } catch (error) {
          reject(new Error(`Failed to parse API response: ${error.message}`));
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

// ===== MAIN EXECUTION =====
async function main() {
  const startTime = Date.now();
  console.log('\\n╔════════════════════════════════════════════════════════╗');
  console.log('║  📊 TOP 10 - 4 SAATLİK SİNYALLER - BAŞLADI          ║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log(`⏰ Başlangıç: ${new Date().toLocaleString('tr-TR')}\\n`);

  try {
    // 1. Fetch TOP 10 signals from API
    console.log('📡 TOP 10 Signals API\\'ye bağlanılıyor...');
    const signalsData = await fetchTop10Signals();

    console.log(`✅ API yanıt verdi:`);
    console.log(`   - Taranan coin sayısı: ${signalsData.totalCoinsScanned}`);
    console.log(`   - İşlem süresi: ${signalsData.elapsedTimeMs}ms`);
    console.log(`   - Zaman damgası: ${signalsData.timestamp}`);
    console.log(`   - AL sinyali sayısı: ${signalsData.top10Coins.filter(c => c.signal === 'AL').length}`);
    console.log(`   - BEKLE sinyali sayısı: ${signalsData.top10Coins.filter(c => c.signal === 'BEKLE').length}\\n`);

    // 2. Get Telegram messages
    const summaryMessage = signalsData.summaryMessage;
    const detailedMessages = signalsData.detailedMessages;

    if (!summaryMessage || !detailedMessages || detailedMessages.length === 0) {
      throw new Error('Telegram messages not found in API response');
    }

    console.log(`📤 Telegram bildirimleri gönderiliyor...`);
    console.log(`   - Özet mesaj: 1 adet`);
    console.log(`   - Detaylı mesajlar: ${detailedMessages.length} adet`);
    console.log(`   - Toplam gönderilecek: ${1 + detailedMessages.length} mesaj/kişi\\n`);

    let successCount = 0;
    let errorCount = 0;
    let totalMessagesSent = 0;
    let trackedSignalsCount = 0;

    // 3. Send to all allowed chat IDs
    for (const chatId of TELEGRAM_CHAT_IDS) {
      console.log(`\\n📤 ${chatId}'e gönderiliyor...`);
      let userSuccessCount = 0;
      let userErrorCount = 0;

      try {
        // A. Send SUMMARY message
        console.log(`   → Özet mesajı gönderiliyor...`);
        await sendTelegramMessage(chatId.trim(), summaryMessage);
        userSuccessCount++;
        totalMessagesSent++;
        console.log(`   ✅ Özet mesajı gönderildi`);

        // Wait 500ms between messages (rate limiting)
        await new Promise((resolve) => setTimeout(resolve, 500));

        // B. Send DETAILED messages (10 messages)
        console.log(`   → ${detailedMessages.length} detaylı mesaj gönderiliyor...`);
        for (let i = 0; i < detailedMessages.length; i++) {
          try {
            await sendTelegramMessage(chatId.trim(), detailedMessages[i]);
            userSuccessCount++;
            totalMessagesSent++;
            console.log(`   ✅ Detaylı mesaj ${i + 1}/${detailedMessages.length} gönderildi`);

            // Track signal in database (only once for first user)
            if (TELEGRAM_CHAT_IDS.indexOf(chatId) === 0 && signalsData.top10Coins[i]) {
              try {
                const coin = signalsData.top10Coins[i];
                await trackSignal({
                  symbol: coin.symbol,
                  signalType: coin.signal,
                  entryPrice: coin.entry,
                  tp1: coin.tp1,
                  tp2: coin.tp2,
                  tp3: coin.tp3,
                  stopLoss: coin.stopLoss,
                  confidence: coin.confidence,
                  source: '4h-top10',
                  reasons: coin.reasons || [],
                  leverage: coin.leverage,
                  capitalAllocation: coin.capitalAllocation
                });
                trackedSignalsCount++;
                console.log(`   📊 Signal tracked: ${coin.symbol} (${coin.signal})`);
              } catch (trackError) {
                console.error(`   ⚠️ Signal tracking hatası: ${trackError.message}`);
              }
            }

            // Wait 500ms between each detailed message
            if (i < detailedMessages.length - 1) {
              await new Promise((resolve) => setTimeout(resolve, 500));
            }
          } catch (error) {
            userErrorCount++;
            console.error(`   ❌ Detaylı mesaj ${i + 1} gönderilemedi: ${error.message}`);
          }
        }

        console.log(`   ✅ ${chatId}: ${userSuccessCount} başarılı, ${userErrorCount} hatalı`);
        successCount++;

      } catch (error) {
        errorCount++;
        console.error(`   ❌ ${chatId}'e gönderilemedi: ${error.message}`);
      }

      // Wait 1 second between users (rate limiting)
      if (TELEGRAM_CHAT_IDS.indexOf(chatId) < TELEGRAM_CHAT_IDS.length - 1) {
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    }

    const totalTime = Date.now() - startTime;

    console.log('\\n╔════════════════════════════════════════════════════════╗');
    console.log('║  ✅ TOP 10 - 4 SAATLİK SİNYALLER - TAMAMLANDI       ║');
    console.log('╚════════════════════════════════════════════════════════╝');
    console.log(`📊 Özet:`);
    console.log(`   - Başarılı kullanıcı: ${successCount}/${TELEGRAM_CHAT_IDS.length}`);
    console.log(`   - Başarısız kullanıcı: ${errorCount}/${TELEGRAM_CHAT_IDS.length}`);
    console.log(`   - Toplam mesaj gönderildi: ${totalMessagesSent}`);
    console.log(`   - Tracking: ${trackedSignalsCount} sinyal kaydedildi`);
    console.log(`   - Toplam süre: ${totalTime}ms (${(totalTime / 1000).toFixed(1)}s)`);
    console.log(`⏰ Bitiş: ${new Date().toLocaleString('tr-TR')}\\n`);

    // Exit successfully
    process.exit(0);
  } catch (error) {
    console.error('\\n╔════════════════════════════════════════════════════════╗');
    console.error('║  ❌ HATA OLUŞTU!                                      ║');
    console.error('╚════════════════════════════════════════════════════════╝');
    console.error(`❌ Hata detayı: ${error.message}`);
    console.error(`📍 Stack: ${error.stack}\\n`);

    // Try to send error notification to first chat ID
    try {
      const errorMessage = `⚠️ *TOP 10 4H SİNYAL SİSTEMİ HATASI*\\n\\n` +
        `❌ Hata: ${error.message}\\n` +
        `⏰ Zaman: ${new Date().toLocaleString('tr-TR')}\\n\\n` +
        `_Sistem yöneticisi bilgilendirildi._`;

      await sendTelegramMessage(TELEGRAM_CHAT_IDS[0].trim(), errorMessage);
    } catch (notificationError) {
      console.error(`❌ Hata bildirimi gönderilemedi: ${notificationError.message}`);
    }

    // Exit with error
    process.exit(1);
  }
}

// ===== RUN =====
if (require.main === module) {
  main();
}

module.exports = { main, fetchTop10Signals, sendTelegramMessage };
