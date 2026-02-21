/**
 * 📈 PERFORMANCE CALCULATOR SERVICE
 *
 * Her gün günlük performans raporunu hesaplayıp Telegram'a gönderir:
 * - Bugünkü sinyal sayısı ve doğruluk oranı
 * - Haftalık ve aylık istatistikler
 * - Kaynak bazlı breakdown (premium, market-briefing, 4h-top10)
 * - Ortalama PnL
 * - En iyi ve en kötü performans gösteren coin'ler
 *
 * PM2 Cron: Her gün 23:59'da çalıştır (Türkiye saati)
 * pm2 start performance-calculator.js --cron "59 23 * * *"
 */

const https = require('https');
const fs = require('fs');
const path = require('path');
const { getStatistics } = require('./signal-tracker');

// ===== CONFIGURATION =====
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || '8292640150:AAHqDdkHxFqx9q8hJ-bJ8KS_Z2LZWrOLroI';
const CONFIG_FILE = path.join(__dirname, '../config.json');
const TRACKING_FILE = path.join(__dirname, '../tracking/signal-history.json');

// ===== GET ADMIN CHAT ID =====
function getAdminChatId() {
  try {
    if (fs.existsSync(CONFIG_FILE)) {
      const config = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf8'));
      return config.adminChatId;
    }
  } catch (error) {
    console.error(`⚠️ Config dosyası okunamadı: ${error.message}`);
  }

  // Fallback to env variable
  return process.env.TELEGRAM_ADMIN_CHAT_ID || '7575640489';
}

// ===== TELEGRAM API =====
async function sendTelegramMessage(chatId, message) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      chat_id: chatId,
      text: message,
      parse_mode: 'Markdown',
      disable_web_page_preview: true,
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

// ===== GET TOP PERFORMERS =====
/**
 * En iyi ve en kötü performans gösteren coin'leri bul
 * @returns {Object} - { topWinners, topLosers }
 */
function getTopPerformers() {
  try {
    if (!fs.existsSync(TRACKING_FILE)) {
      return { topWinners: [], topLosers: [] };
    }

    const fileContent = fs.readFileSync(TRACKING_FILE, 'utf8');
    const trackingDb = JSON.parse(fileContent);

    // Bugünkü tamamlanmış sinyaller
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const todaySignals = trackingDb.signals.filter(signal => {
      const signalDate = new Date(signal.timestamp);
      signalDate.setHours(0, 0, 0, 0);
      return signalDate.getTime() === today.getTime() && signal.pnl !== null;
    });

    if (todaySignals.length === 0) {
      return { topWinners: [], topLosers: [] };
    }

    // PnL'ye göre sırala
    const sorted = [...todaySignals].sort((a, b) => b.pnl - a.pnl);

    const topWinners = sorted.slice(0, 3).map(s => ({
      symbol: s.symbol,
      pnl: s.pnl,
      signalType: s.signalType
    }));

    const topLosers = sorted.slice(-3).reverse().map(s => ({
      symbol: s.symbol,
      pnl: s.pnl,
      signalType: s.signalType
    }));

    return { topWinners, topLosers };

  } catch (error) {
    console.error(`❌ Top performers hesaplama hatası: ${error.message}`);
    return { topWinners: [], topLosers: [] };
  }
}

// ===== GENERATE PERFORMANCE REPORT =====
/**
 * Performans raporunu Markdown formatında oluştur
 * @returns {string} - Telegram mesajı
 */
function generatePerformanceReport() {
  // Bugün, Bu Hafta, Bu Ay istatistiklerini al
  const todayStats = getStatistics({ period: 'today' });
  const weekStats = getStatistics({ period: 'week' });
  const monthStats = getStatistics({ period: 'month' });

  // Kaynak bazlı istatistikler
  const premiumStats = getStatistics({ period: 'today', source: 'premium-signals' });
  const marketBriefingStats = getStatistics({ period: 'today', source: 'market-briefing' });
  const top10Stats = getStatistics({ period: 'today', source: '4h-top10' });

  // Top performers
  const { topWinners, topLosers } = getTopPerformers();

  // Rapor oluştur
  const date = new Date().toLocaleDateString('tr-TR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  let report = `📊 *GÜNLÜK PERFORMANS RAPORU*\n`;
  report += `📅 ${date}\n`;
  report += `\n`;
  report += `╔════════════════════════════════╗\n`;
  report += `║  📈 BUGÜN                      ║\n`;
  report += `╚════════════════════════════════╝\n`;
  report += `🎯 *Doğruluk:* ${todayStats.accuracy}%\n`;
  report += `📊 *Toplam Sinyal:* ${todayStats.totalSignals}\n`;
  report += `✅ *Başarılı:* ${todayStats.successfulSignals}\n`;
  report += `❌ *Başarısız:* ${todayStats.failedSignals}\n`;
  report += `⏳ *Beklemede:* ${todayStats.pendingSignals}\n`;
  report += `💰 *Ortalama PnL:* ${todayStats.avgPnl > 0 ? '+' : ''}${todayStats.avgPnl}%\n`;
  report += `\n`;

  // Kaynak bazlı breakdown
  if (todayStats.totalSignals > 0) {
    report += `📋 *Kaynak Bazlı Performans:*\n`;

    if (premiumStats.totalSignals > 0) {
      report += `   • Premium Signals: ${premiumStats.accuracy}% (${premiumStats.totalSignals} sinyal)\n`;
    }
    if (marketBriefingStats.totalSignals > 0) {
      report += `   • Market Briefing: ${marketBriefingStats.accuracy}% (${marketBriefingStats.totalSignals} sinyal)\n`;
    }
    if (top10Stats.totalSignals > 0) {
      report += `   • 4H TOP 10: ${top10Stats.accuracy}% (${top10Stats.totalSignals} sinyal)\n`;
    }
    report += `\n`;
  }

  // Top performers (bugün)
  if (topWinners.length > 0) {
    report += `🏆 *En İyi 3 Coin (Bugün):*\n`;
    topWinners.forEach((coin, i) => {
      const emoji = i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉';
      report += `   ${emoji} ${coin.symbol}: ${coin.pnl > 0 ? '+' : ''}${coin.pnl.toFixed(2)}%\n`;
    });
    report += `\n`;
  }

  if (topLosers.length > 0 && topLosers[0].pnl < 0) {
    report += `⚠️ *En Kötü 3 Coin (Bugün):*\n`;
    topLosers.forEach(coin => {
      report += `   🔻 ${coin.symbol}: ${coin.pnl.toFixed(2)}%\n`;
    });
    report += `\n`;
  }

  // Haftalık ve aylık özet
  report += `╔════════════════════════════════╗\n`;
  report += `║  📆 HAFTALIK & AYLIK ÖZET      ║\n`;
  report += `╚════════════════════════════════╝\n`;
  report += `📅 *Bu Hafta:*\n`;
  report += `   • Doğruluk: ${weekStats.accuracy}%\n`;
  report += `   • Toplam: ${weekStats.totalSignals} sinyal (✅${weekStats.successfulSignals} / ❌${weekStats.failedSignals})\n`;
  report += `   • Ortalama PnL: ${weekStats.avgPnl > 0 ? '+' : ''}${weekStats.avgPnl}%\n`;
  report += `\n`;
  report += `📅 *Bu Ay:*\n`;
  report += `   • Doğruluk: ${monthStats.accuracy}%\n`;
  report += `   • Toplam: ${monthStats.totalSignals} sinyal (✅${monthStats.successfulSignals} / ❌${monthStats.failedSignals})\n`;
  report += `   • Ortalama PnL: ${monthStats.avgPnl > 0 ? '+' : ''}${monthStats.avgPnl}%\n`;
  report += `\n`;

  // Footer
  report += `_🤖 LyDian Trading Bot - Performance Tracking System_\n`;
  report += `_⏰ Rapor Saati: ${new Date().toLocaleTimeString('tr-TR')}_`;

  return report;
}

// ===== MAIN EXECUTION =====
async function main() {
  const startTime = Date.now();
  console.log('\n╔════════════════════════════════════════════════════════╗');
  console.log('║  📈 PERFORMANCE CALCULATOR - BAŞLADI                  ║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log(`⏰ Başlangıç: ${new Date().toLocaleString('tr-TR')}\n`);

  try {
    // 1. Admin Chat ID'yi al
    const adminChatId = getAdminChatId();
    console.log(`✅ Admin Chat ID: ${adminChatId}`);

    // 2. Performans raporunu oluştur
    console.log('📊 Performans raporu hesaplanıyor...');
    const report = generatePerformanceReport();

    console.log('✅ Rapor hazırlandı.\n');
    console.log('─────────────────────────────────────────────────────');
    console.log(report);
    console.log('─────────────────────────────────────────────────────\n');

    // 3. Telegram'a gönder
    console.log(`📤 Admin'e Telegram mesajı gönderiliyor...`);
    await sendTelegramMessage(adminChatId, report);
    console.log('✅ Rapor başarıyla gönderildi!\n');

    const totalTime = Date.now() - startTime;

    console.log('╔════════════════════════════════════════════════════════╗');
    console.log('║  ✅ PERFORMANCE CALCULATOR - TAMAMLANDI               ║');
    console.log('╚════════════════════════════════════════════════════════╝');
    console.log(`📊 Özet:`);
    console.log(`   - Rapor oluşturuldu ve gönderildi`);
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

module.exports = { main, generatePerformanceReport, getTopPerformers };
