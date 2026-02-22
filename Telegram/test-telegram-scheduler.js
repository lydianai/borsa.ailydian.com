#!/usr/bin/env node
/**
 * 🧪 TELEGRAM SCHEDULER TEST
 * Basit test scripti - TypeScript import sorunlarını bypass eder
 */

const https = require('https');
const http = require('http');

const BASE_URL = 'http://localhost:3000';

// Basit HTTP GET request fonksiyonu
function httpGet(url) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    protocol.get(url, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          resolve({ error: 'Parse error', raw: data });
        }
      });
    }).on('error', reject);
  });
}

async function testAllAPIs() {
  console.log('\n🧪 === TELEGRAM SCHEDULER API TEST ===\n');

  const tests = [
    {
      name: '1. Nirvana Dashboard',
      url: `${BASE_URL}/api/nirvana`,
      check: (data) => data.success && data.data?.totalStrategies > 0,
    },
    {
      name: '2. Omnipotent Futures',
      url: `${BASE_URL}/api/omnipotent-futures?limit=10`,
      check: (data) => data.success && data.data?.futures?.length > 0,
    },
    {
      name: '3. BTC-ETH Analysis',
      url: `${BASE_URL}/api/btc-eth-analysis`,
      check: (data) => data.success && data.data?.correlation30d !== undefined,
    },
    {
      name: '4. Market Correlation',
      url: `${BASE_URL}/api/market-correlation?limit=10`,
      check: (data) => data.success && data.data?.correlations?.length > 0,
    },
    {
      name: '5. Crypto News (Groq)',
      url: `${BASE_URL}/api/crypto-news?refresh=true`,
      check: (data) => data.success && data.data?.length > 0,
    },
  ];

  let passed = 0;
  let failed = 0;

  for (const test of tests) {
    try {
      process.stdout.write(`${test.name}... `);
      const data = await httpGet(test.url);

      if (test.check(data)) {
        console.log('✅ PASS');
        passed++;
      } else {
        console.log('❌ FAIL -', JSON.stringify(data).substring(0, 100));
        failed++;
      }
    } catch (error) {
      console.log('❌ ERROR -', error.message);
      failed++;
    }
  }

  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`📊 Sonuç: ${passed} PASS / ${failed} FAIL`);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  if (failed === 0) {
    console.log('🎉 TÜM TESTLERden GEÇTI!\n');
    console.log('✨ Telegram Scheduler kullanıma hazır!\n');
    console.log('🚀 Başlatmak için:');
    console.log('   cd /Users/lydian/Documents/lytrade-final.bak-20251030-170900/Telegram/schedulers');
    console.log('   pm2 start ecosystem.config.js\n');
  } else {
    console.log('⚠️  Bazı testler başarısız oldu. Lütfen API\'leri kontrol edin.\n');
    process.exit(1);
  }
}

// Test'i çalıştır
testAllAPIs().catch(err => {
  console.error('\n❌ Test hatası:', err);
  process.exit(1);
});
