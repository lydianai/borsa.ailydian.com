/**
 * 📊 SAAT BAŞI PİYASA BİLGİLENDİRME SERVİSİ
 *
 * Trader'lar için kritik piyasa bilgilerini her saat başı Telegram'a gönderir:
 * - Global piyasa durumu
 * - Fear & Greed Index
 * - En çok yükselen/düşen coinler
 * - BTC/ETH durumu
 * - Önemli uyarılar
 *
 * PM2 ile çalıştırılır: pm2 start hourly-market-briefing.js --cron "0 * * * *"
 */

const https = require('https');
const http = require('http');
const fs = require('fs');
const path = require('path');

// ===== CONFIGURATION =====
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || '8292640150:AAHqDdkHxFqx9q8hJ-bJ8KS_Z2LZWrOLroI';
const SUBSCRIBERS_FILE = path.join(__dirname, '../', 'subscribers.json');
const API_BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3000';

// ===== SUBSCRIBER MANAGEMENT =====
function getActiveSubscribers() {
  try {
    if (fs.existsSync(SUBSCRIBERS_FILE)) {
      const data = JSON.parse(fs.readFileSync(SUBSCRIBERS_FILE, 'utf8'));
      return data.subscribers
        .filter(sub => sub.active)
        .map(sub => sub.chatId.toString());
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

// ===== FETCH MARKET BRIEFING =====
async function fetchMarketBriefing() {
  return new Promise((resolve, reject) => {
    const url = `${API_BASE_URL}/api/telegram/market-briefing`;
    const parsedUrl = new URL(url);

    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port || (parsedUrl.protocol === 'https:' ? 443 : 80),
      path: parsedUrl.pathname + parsedUrl.search,
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 60000, // 1 minute (faster than premium signals)
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
  console.log('\n╔════════════════════════════════════════════════════════╗');
  console.log('║  📊 SAAT BAŞI PİYASA BİLGİLENDİRME - BAŞLADI         ║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log(`⏰ Başlangıç: ${new Date().toLocaleString('tr-TR')}\n`);

  try {
    // 1. Fetch briefing from API
    console.log('📡 Market Briefing API\'ye bağlanılıyor...');
    const briefingData = await fetchMarketBriefing();

    console.log(`✅ API yanıt verdi:`);
    console.log(`   - İşlem süresi: ${briefingData.elapsedTimeMs}ms`);
    console.log(`   - Zaman damgası: ${briefingData.timestamp}\n`);

    // 2. Get Telegram message
    const telegramMessage = briefingData.telegramMessage;

    if (!telegramMessage) {
      throw new Error('Telegram message not found in API response');
    }

    // 3. Send to all allowed chat IDs
    console.log(`📤 Telegram bildirimler gönderiliyor...`);
    let successCount = 0;
    let errorCount = 0;

    for (const chatId of TELEGRAM_CHAT_IDS) {
      try {
        console.log(`   → ${chatId}'e gönderiliyor...`);
        await sendTelegramMessage(chatId.trim(), telegramMessage);
        successCount++;
        console.log(`   ✅ ${chatId}'e başarıyla gönderildi`);
      } catch (error) {
        errorCount++;
        console.error(`   ❌ ${chatId}'e gönderilemedi: ${error.message}`);
      }

      // Rate limiting: wait 100ms between messages
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    const totalTime = Date.now() - startTime;

    console.log('\n╔════════════════════════════════════════════════════════╗');
    console.log('║  ✅ SAAT BAŞI PİYASA BİLGİLENDİRME - TAMAMLANDI      ║');
    console.log('╚════════════════════════════════════════════════════════╝');
    console.log(`📊 Özet:`);
    console.log(`   - Başarılı: ${successCount}/${TELEGRAM_CHAT_IDS.length}`);
    console.log(`   - Başarısız: ${errorCount}/${TELEGRAM_CHAT_IDS.length}`);
    console.log(`   - Toplam süre: ${totalTime}ms`);
    console.log(`⏰ Bitiş: ${new Date().toLocaleString('tr-TR')}\n`);

    // Exit successfully
    process.exit(0);
  } catch (error) {
    console.error('\n╔════════════════════════════════════════════════════════╗');
    console.error('║  ❌ HATA OLUŞTU!                                      ║');
    console.error('╚════════════════════════════════════════════════════════╝');
    console.error(`❌ Hata detayı: ${error.message}`);
    console.error(`📍 Stack: ${error.stack}\n`);

    // Try to send error notification to first chat ID
    try {
      const errorMessage = `⚠️ *PİYASA BİLGİLENDİRME SİSTEMİ HATASI*\n\n` +
        `❌ Hata: ${error.message}\n` +
        `⏰ Zaman: ${new Date().toLocaleString('tr-TR')}\n\n` +
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

module.exports = { main, fetchMarketBriefing, sendTelegramMessage };
