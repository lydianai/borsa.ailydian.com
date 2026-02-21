/**
 * 🤖 TELEGRAM BOT LISTENER
 *
 * Bu servis Telegram bot'a gelen mesajları dinler ve:
 * - /start komutu ile kullanıcıları subscribers listesine ekler
 * - /stop komutu ile kullanıcıları listeden çıkarır
 * - /status komutu ile mevcut abonelik durumunu gösterir
 *
 * PM2 ile sürekli çalışır: pm2 start bot-listener.js --name telegram-bot-listener
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// ===== CONFIGURATION =====
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || '8292640150:AAHqDdkHxFqx9q8hJ-bJ8KS_Z2LZWrOLroI';
const SUBSCRIBERS_FILE = path.join(__dirname, 'subscribers.json');
const POLLING_INTERVAL = 2000; // 2 seconds

// ===== SUBSCRIBER MANAGEMENT =====
function loadSubscribers() {
  try {
    if (fs.existsSync(SUBSCRIBERS_FILE)) {
      const data = fs.readFileSync(SUBSCRIBERS_FILE, 'utf8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error(`❌ Subscribers dosyası okunamadı: ${error.message}`);
  }
  return { subscribers: [], lastUpdate: new Date().toISOString() };
}

function saveSubscribers(data) {
  try {
    data.lastUpdate = new Date().toISOString();
    fs.writeFileSync(SUBSCRIBERS_FILE, JSON.stringify(data, null, 2));
    return true;
  } catch (error) {
    console.error(`❌ Subscribers dosyası kaydedilemedi: ${error.message}`);
    return false;
  }
}

function addSubscriber(chatId, username, firstName, lastName) {
  const data = loadSubscribers();

  const existing = data.subscribers.find(sub => sub.chatId === chatId);
  if (existing) {
    existing.username = username;
    existing.firstName = firstName;
    existing.lastName = lastName;
    existing.lastActive = new Date().toISOString();
    existing.active = true;
    saveSubscribers(data);
    return { alreadySubscribed: true, subscriber: existing };
  }

  const subscriber = {
    chatId,
    username: username || 'N/A',
    firstName: firstName || 'Anonim',
    lastName: lastName || '',
    subscribedAt: new Date().toISOString(),
    lastActive: new Date().toISOString(),
    active: true,
  };

  data.subscribers.push(subscriber);
  saveSubscribers(data);

  console.log(`✅ Yeni abone eklendi: ${chatId} (${subscriber.firstName})`);
  return { alreadySubscribed: false, subscriber };
}

function removeSubscriber(chatId) {
  const data = loadSubscribers();
  const subscriber = data.subscribers.find(sub => sub.chatId === chatId);

  if (subscriber) {
    subscriber.active = false;
    subscriber.unsubscribedAt = new Date().toISOString();
    saveSubscribers(data);
    console.log(`❌ Abone çıkarıldı: ${chatId}`);
    return true;
  }

  return false;
}

function getActiveSubscribers() {
  const data = loadSubscribers();
  return data.subscribers.filter(sub => sub.active);
}

// ===== TELEGRAM API =====
async function sendMessage(chatId, text, options = {}) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify({
      chat_id: chatId,
      text: text,
      parse_mode: 'Markdown',
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
            reject(new Error(`Telegram API error: ${res.statusCode}`));
          }
        });
      }
    );

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function getUpdates(offset = 0) {
  return new Promise((resolve, reject) => {
    const req = https.request(
      {
        hostname: 'api.telegram.org',
        port: 443,
        path: `/bot${TELEGRAM_BOT_TOKEN}/getUpdates?offset=${offset}&timeout=30`,
        method: 'GET',
      },
      (res) => {
        let data = '';
        res.on('data', (chunk) => {
          data += chunk;
        });
        res.on('end', () => {
          try {
            const result = JSON.parse(data);
            if (result.ok) {
              resolve(result.result);
            } else {
              reject(new Error(result.description || 'Unknown error'));
            }
          } catch (error) {
            reject(error);
          }
        });
      }
    );

    req.on('error', reject);
    req.end();
  });
}

// ===== COMMAND HANDLERS =====
async function handleStartCommand(chatId, user) {
  const result = addSubscriber(
    chatId,
    user.username,
    user.first_name,
    user.last_name
  );

  let message;
  if (result.alreadySubscribed) {
    message = `✅ *Tekrar Hoşgeldiniz!*\n\n` +
      `Zaten abone listemizdeysiniz ${result.subscriber.firstName}!\n\n` +
      `📊 *Alacağınız Bildirimler:*\n` +
      `• 🔥 Saatlik Premium Sinyaller\n` +
      `• 📊 Saatlik Piyasa Bilgilendirmesi\n` +
      `• ⚠️ Kritik Piyasa Uyarıları\n\n` +
      `📱 Bildirimler otomatik olarak her saat başı gelecek.\n\n` +
      `_Komutlar:_\n` +
      `/stop - Bildirimleri durdur\n` +
      `/status - Abonelik durumunu gör`;
  } else {
    message = `🎉 *Hoşgeldiniz ${result.subscriber.firstName}!*\n\n` +
      `✅ Başarıyla abone oldunuz!\n\n` +
      `📊 *Alacağınız Bildirimler:*\n` +
      `• 🔥 Saatlik Premium Sinyaller (Giriş/TP/SL dahil)\n` +
      `• 📊 Saatlik Piyasa Bilgilendirmesi\n` +
      `• ⚠️ Önemli Piyasa Hareketleri\n\n` +
      `⏰ Bildirimler her saat başı otomatik gönderilecek.\n\n` +
      `⚠️ *DİKKAT:* Bu sinyaller sadece bilgilendirme amaçlıdır. Yatırım kararlarınızı kendiniz verin.\n\n` +
      `_Komutlar:_\n` +
      `/stop - Bildirimleri durdur\n` +
      `/status - Abonelik durumunu gör`;
  }

  await sendMessage(chatId, message);
}

async function handleStopCommand(chatId, user) {
  const removed = removeSubscriber(chatId);

  let message;
  if (removed) {
    message = `👋 *Görüşmek Üzere!*\n\n` +
      `Bildirimleriniz durduruldu.\n\n` +
      `Tekrar abone olmak isterseniz `/start` komutunu kullanabilirsiniz.\n\n` +
      `_İyi günler dileriz!_ 🙏`;
  } else {
    message = `⚠️ Zaten abone değilsiniz.\n\n` +
      `Abone olmak için `/start` komutunu kullanın.`;
  }

  await sendMessage(chatId, message);
}

async function handleStatusCommand(chatId) {
  const data = loadSubscribers();
  const subscriber = data.subscribers.find(sub => sub.chatId === chatId);

  let message;
  if (subscriber && subscriber.active) {
    const activeCount = getActiveSubscribers().length;
    const subscribedDate = new Date(subscriber.subscribedAt).toLocaleString('tr-TR');

    message = `📊 *Abonelik Durumunuz*\n\n` +
      `✅ Aktif Abone\n\n` +
      `👤 Ad: ${subscriber.firstName} ${subscriber.lastName}\n` +
      `📅 Abone Olma: ${subscribedDate}\n` +
      `👥 Toplam Aktif Abone: ${activeCount}\n\n` +
      `📱 *Aldığınız Bildirimler:*\n` +
      `• 🔥 Premium Sinyaller (Saatlik)\n` +
      `• 📊 Piyasa Bilgilendirmesi (Saatlik)\n\n` +
      `_Komutlar:_\n` +
      `/stop - Bildirimleri durdur`;
  } else {
    message = `❌ *Aktif Abone Değilsiniz*\n\n` +
      `Bildirim almak için abone olun:\n` +
      `/start - Abone ol`;
  }

  await sendMessage(chatId, message);
}

async function handleHelpCommand(chatId) {
  const message = `📖 *Yardım Menüsü*\n\n` +
    `*Kullanılabilir Komutlar:*\n\n` +
    `/start - Bildirimlere abone ol\n` +
    `/stop - Bildirimleri durdur\n` +
    `/status - Abonelik durumunu gör\n` +
    `/help - Bu yardım mesajını göster\n\n` +
    `📊 *Hakkında:*\n` +
    `Bu bot saatlik crypto trading sinyalleri ve piyasa bilgilendirmeleri gönderir.\n\n` +
    `• 18+ strateji konsensüsü\n` +
    `• Gerçek zamanlı fiyat analizi\n` +
    `• Whale aktivitesi tespiti\n` +
    `• TP/SL/Giriş seviyeleri\n\n` +
    `⚠️ *Feragatname:* Bu bildirimler sadece bilgilendirme amaçlıdır. Yatırım tavsiyesi değildir.`;

  await sendMessage(chatId, message);
}

async function handleUnknownCommand(chatId) {
  const message = `❓ *Bilinmeyen Komut*\n\n` +
    `Kullanılabilir komutlar:\n\n` +
    `/start - Abone ol\n` +
    `/stop - Abonelikten çık\n` +
    `/status - Durum bilgisi\n` +
    `/help - Yardım`;

  await sendMessage(chatId, message);
}

// ===== UPDATE PROCESSOR =====
async function processUpdate(update) {
  try {
    if (!update.message || !update.message.text) {
      return;
    }

    const chatId = update.message.chat.id;
    const text = update.message.text.trim();
    const user = update.message.from;

    console.log(`📨 Mesaj alındı: ${text} (${chatId})`);

    // Command handling
    if (text === '/start') {
      await handleStartCommand(chatId, user);
    } else if (text === '/stop') {
      await handleStopCommand(chatId, user);
    } else if (text === '/status') {
      await handleStatusCommand(chatId);
    } else if (text === '/help') {
      await handleHelpCommand(chatId);
    } else if (text.startsWith('/')) {
      await handleUnknownCommand(chatId);
    } else {
      // Non-command message - send help
      await sendMessage(
        chatId,
        `Merhaba! Ben bir sinyal bot'uyum. 🤖\n\nAbone olmak için /start komutunu kullanın.`
      );
    }
  } catch (error) {
    console.error(`❌ Update işlenirken hata: ${error.message}`);
  }
}

// ===== MAIN POLLING LOOP =====
let lastUpdateId = 0;
let isRunning = true;

async function poll() {
  while (isRunning) {
    try {
      const updates = await getUpdates(lastUpdateId + 1);

      for (const update of updates) {
        await processUpdate(update);
        lastUpdateId = update.update_id;
      }

      // Short delay between polls
      await new Promise(resolve => setTimeout(resolve, 100));
    } catch (error) {
      console.error(`❌ Polling hatası: ${error.message}`);
      // Wait longer before retrying after an error
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
}

// ===== STARTUP =====
async function start() {
  console.log('\n╔════════════════════════════════════════════════════════╗');
  console.log('║  🤖 TELEGRAM BOT LISTENER - BAŞLATILDI               ║');
  console.log('╚════════════════════════════════════════════════════════╝');
  console.log(`⏰ Başlangıç: ${new Date().toLocaleString('tr-TR')}\n`);

  // Initialize subscribers file if it doesn't exist
  if (!fs.existsSync(SUBSCRIBERS_FILE)) {
    saveSubscribers({ subscribers: [] });
    console.log('✅ Subscribers dosyası oluşturuldu');
  }

  const activeSubscribers = getActiveSubscribers();
  console.log(`👥 Aktif abone sayısı: ${activeSubscribers.length}\n`);

  console.log('🔄 Polling başlatılıyor...\n');

  // Start polling
  poll().catch(error => {
    console.error(`❌ Fatal error: ${error.message}`);
    process.exit(1);
  });
}

// ===== GRACEFUL SHUTDOWN =====
process.on('SIGINT', () => {
  console.log('\n\n🛑 SIGINT alındı, kapatılıyor...');
  isRunning = false;
  setTimeout(() => {
    console.log('👋 Bot listener durduruldu\n');
    process.exit(0);
  }, 1000);
});

process.on('SIGTERM', () => {
  console.log('\n\n🛑 SIGTERM alındı, kapatılıyor...');
  isRunning = false;
  setTimeout(() => {
    console.log('👋 Bot listener durduruldu\n');
    process.exit(0);
  }, 1000);
});

// ===== RUN =====
if (require.main === module) {
  start();
}

module.exports = {
  loadSubscribers,
  saveSubscribers,
  addSubscriber,
  removeSubscriber,
  getActiveSubscribers,
};
