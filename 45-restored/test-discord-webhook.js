#!/usr/bin/env node

/**
 * DISCORD WEBHOOK TEST SCRIPT
 * Test Discord webhook integration
 */

require('dotenv').config();

const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

async function testDiscordWebhook() {
  console.log('💬 Discord Webhook Test Başlıyor...\n');

  // Check env variables
  console.log('📋 Environment Variables:');
  console.log(`DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL ? '✅ Mevcut' : '❌ YOK'}\n`);

  if (!DISCORD_WEBHOOK_URL) {
    console.error('❌ HATA: Discord webhook URL .env dosyasında bulunamadı!');
    console.log('\n📚 Setup Rehberi: DISCORD-WEBHOOK-SETUP-GUIDE.md\n');
    process.exit(1);
  }

  if (DISCORD_WEBHOOK_URL === 'your_discord_webhook_url_here') {
    console.error('❌ HATA: Placeholder değer değiştirilmemiş!');
    console.log('\n📚 Setup Rehberi: DISCORD-WEBHOOK-SETUP-GUIDE.md\n');
    process.exit(1);
  }

  // Test 1: Simple message
  console.log('📤 Test 1: Basit mesaj gönderiliyor...');
  try {
    const response1 = await fetch(DISCORD_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: '🚀 Test Mesajı!\n\nDiscord webhook başarıyla çalışıyor!',
      }),
    });

    if (response1.ok) {
      console.log('✅ Test 1 BAŞARILI: Basit mesaj gönderildi');
    } else {
      const error = await response1.text();
      console.error('❌ Test 1 BAŞARISIZ:', response1.status, error);
    }
  } catch (error) {
    console.error('❌ Test 1 HATA:', error.message);
  }

  // Wait 1 second
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Test 2: Embed message (CRITICAL)
  console.log('\n📤 Test 2: Critical alert (embed format)...');
  try {
    const response2 = await fetch(DISCORD_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        embeds: [
          {
            title: '🚨 Emergency Stop Activated',
            description: 'Bot stopped due to emergency condition',
            color: 16711680, // Red
            timestamp: new Date().toISOString(),
            footer: {
              text: 'Severity: CRITICAL',
            },
            fields: [
              { name: 'Bot', value: 'BTCUSDT Quantum Bot', inline: true },
              { name: 'Action', value: 'All positions closed', inline: true },
            ],
          },
        ],
      }),
    });

    if (response2.ok) {
      console.log('✅ Test 2 BAŞARILI: Critical alert (embed) gönderildi');
    } else {
      const error = await response2.text();
      console.error('❌ Test 2 BAŞARISIZ:', response2.status, error);
    }
  } catch (error) {
    console.error('❌ Test 2 HATA:', error.message);
  }

  // Wait 1 second
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Test 3: Trading alert (SUCCESS)
  console.log('\n📤 Test 3: Trading success alert...');
  try {
    const response3 = await fetch(DISCORD_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        embeds: [
          {
            title: '📊 Position Closed - Profit',
            description: '✅ LONG BTCUSDT successfully closed',
            color: 65280, // Green
            timestamp: new Date().toISOString(),
            footer: {
              text: 'Severity: SUCCESS',
            },
            fields: [
              { name: 'Entry', value: '$42,150', inline: true },
              { name: 'Exit', value: '$42,395', inline: true },
              { name: 'P&L', value: '+245.50 USDT', inline: true },
              { name: 'Win Rate', value: '68.5%', inline: true },
              { name: 'Sharpe Ratio', value: '2.45', inline: true },
              { name: 'Duration', value: '4h 23m', inline: true },
            ],
          },
        ],
      }),
    });

    if (response3.ok) {
      console.log('✅ Test 3 BAŞARILI: Success alert (embed) gönderildi');
    } else {
      const error = await response3.text();
      console.error('❌ Test 3 BAŞARISIZ:', response3.status, error);
    }
  } catch (error) {
    console.error('❌ Test 3 HATA:', error.message);
  }

  // Wait 1 second
  await new Promise(resolve => setTimeout(resolve, 1000));

  // Test 4: Warning alert (HIGH)
  console.log('\n📤 Test 4: High severity warning...');
  try {
    const response4 = await fetch(DISCORD_WEBHOOK_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        embeds: [
          {
            title: '⚠️ Maximum Drawdown Warning',
            description: 'Approaching maximum allowed drawdown',
            color: 16750848, // Orange
            timestamp: new Date().toISOString(),
            footer: {
              text: 'Severity: HIGH',
            },
            fields: [
              { name: 'Current Drawdown', value: '18.2%', inline: true },
              { name: 'Max Allowed', value: '20.0%', inline: true },
              { name: 'Remaining', value: '1.8%', inline: true },
            ],
          },
        ],
      }),
    });

    if (response4.ok) {
      console.log('✅ Test 4 BAŞARILI: Warning alert (embed) gönderildi');
    } else {
      const error = await response4.text();
      console.error('❌ Test 4 BAŞARISIZ:', response4.status, error);
    }
  } catch (error) {
    console.error('❌ Test 4 HATA:', error.message);
  }

  console.log('\n✅ TEST TAMAMLANDI!\n');
  console.log('💬 Discord kanalını kontrol et - 4 mesaj göreceksin:');
  console.log('   1. Basit test mesajı');
  console.log('   2. Critical alert (kırmızı embed)');
  console.log('   3. Success alert (yeşil embed)');
  console.log('   4. Warning alert (turuncu embed)');
  console.log('\n🔥 Sonraki Adım: API üzerinden alert test et:');
  console.log('   curl -X POST http://localhost:3000/api/monitoring/live \\');
  console.log('     -H "Content-Type: application/json" \\');
  console.log('     -d \'{"action":"emergency_stop"}\'');
  console.log('');
}

testDiscordWebhook().catch(console.error);
