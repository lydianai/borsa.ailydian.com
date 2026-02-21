#!/usr/bin/env node

/**
 * TELEGRAM ALERT TEST SCRIPT
 * Test Telegram bot integration
 */

require('dotenv').config();

const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID;

async function testTelegramBot() {
  console.log('🤖 Telegram Bot Test Başlıyor...\n');

  // Check env variables
  console.log('📋 Environment Variables:');
  console.log(`TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN ? '✅ Mevcut' : '❌ YOK'}`);
  console.log(`TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID ? '✅ Mevcut' : '❌ YOK'}\n`);

  if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
    console.error('❌ HATA: Telegram credentials .env dosyasında bulunamadı!');
    console.log('\n📚 Setup Rehberi: TELEGRAM-BOT-SETUP-GUIDE.md\n');
    process.exit(1);
  }

  if (
    TELEGRAM_BOT_TOKEN === 'your_telegram_bot_token_here' ||
    TELEGRAM_CHAT_ID === 'your_telegram_chat_id_here'
  ) {
    console.error('❌ HATA: Placeholder değerleri değiştirilmemiş!');
    console.log('\n📚 Setup Rehberi: TELEGRAM-BOT-SETUP-GUIDE.md\n');
    process.exit(1);
  }

  // Test 1: Simple message
  console.log('📤 Test 1: Basit mesaj gönderiliyor...');
  try {
    const response1 = await fetch(
      `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text: '🚀 Test Mesajı!\n\nTelegram bot başarıyla çalışıyor!',
        }),
      }
    );

    const result1 = await response1.json();

    if (result1.ok) {
      console.log('✅ Test 1 BAŞARILI: Basit mesaj gönderildi');
    } else {
      console.error('❌ Test 1 BAŞARISIZ:', result1.description);
    }
  } catch (error) {
    console.error('❌ Test 1 HATA:', error.message);
  }

  // Test 2: Formatted message (Markdown)
  console.log('\n📤 Test 2: Markdown formatında mesaj...');
  try {
    const response2 = await fetch(
      `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text: `🚨 *CRITICAL ALERT*\n\n⚠️ Maximum drawdown exceeded!\n\n_${new Date().toLocaleString()}_`,
          parse_mode: 'Markdown',
        }),
      }
    );

    const result2 = await response2.json();

    if (result2.ok) {
      console.log('✅ Test 2 BAŞARILI: Markdown mesaj gönderildi');
    } else {
      console.error('❌ Test 2 BAŞARISIZ:', result2.description);
    }
  } catch (error) {
    console.error('❌ Test 2 HATA:', error.message);
  }

  // Test 3: Trading alert simulation
  console.log('\n📤 Test 3: Trading alert simülasyonu...');
  try {
    const response3 = await fetch(
      `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          chat_id: TELEGRAM_CHAT_ID,
          text: `📊 *Position Closed*\n\n✅ LONG BTCUSDT\n💰 P&L: +245.50 USDT\n📈 Win Rate: 68.5%\n\n_${new Date().toLocaleString()}_`,
          parse_mode: 'Markdown',
        }),
      }
    );

    const result3 = await response3.json();

    if (result3.ok) {
      console.log('✅ Test 3 BAŞARILI: Trading alert gönderildi');
    } else {
      console.error('❌ Test 3 BAŞARISIZ:', result3.description);
    }
  } catch (error) {
    console.error('❌ Test 3 HATA:', error.message);
  }

  console.log('\n✅ TEST TAMAMLANDI!\n');
  console.log('📱 Telegram\'ı kontrol et - 3 mesaj göreceksin.');
  console.log('\n🔥 Sonraki Adım: Emergency stop alert test et:');
  console.log('   curl -X POST http://localhost:3000/api/monitoring/live \\');
  console.log('     -H "Content-Type: application/json" \\');
  console.log('     -d \'{"action":"emergency_stop"}\'');
  console.log('');
}

testTelegramBot().catch(console.error);
