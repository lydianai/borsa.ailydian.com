#!/usr/bin/env node

/**
 * BOT INTEGRATION TEST SCRIPT
 * Test real bot initialization and control
 */

const API_URL = 'http://localhost:3000';

// Test configuration
const TEST_CONFIG = {
  apiKey: 'test_api_key_binance', // TESTNET için placeholder
  apiSecret: 'test_api_secret_binance', // TESTNET için placeholder
  config: {
    symbol: 'BTCUSDT',
    leverage: 10,
    maxPositionSize: 100, // USDT
    stopLossPercent: 2,
    takeProfitPercent: 3,
    maxDailyLoss: 50, // USDT
    riskPerTrade: 1, // %
  },
  testnet: true, // TESTNET modunda çalış
};

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function testBotIntegration() {
  console.log('🤖 Bot Integration Test Başlıyor...\n');

  // Test 1: Check initialization status
  console.log('📋 Test 1: Bot initialization status kontrolü...');
  try {
    const statusResponse = await fetch(`${API_URL}/api/bot/initialize`);
    const statusResult = await statusResponse.json();

    if (statusResult.success) {
      console.log('✅ Status check başarılı');
      console.log(`   Initialized: ${statusResult.isInitialized}`);
      if (statusResult.config) {
        console.log(`   Symbol: ${statusResult.config.symbol}`);
        console.log(`   Leverage: ${statusResult.config.leverage}x`);
      }
    } else {
      console.error('❌ Status check başarısız:', statusResult.error);
    }
  } catch (error) {
    console.error('❌ Status check hatası:', error.message);
  }

  await sleep(1000);

  // Test 2: Initialize bot
  console.log('\n🔧 Test 2: Bot initialization...');
  try {
    const initResponse = await fetch(`${API_URL}/api/bot/initialize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(TEST_CONFIG),
    });

    const initResult = await initResponse.json();

    if (initResult.success) {
      console.log('✅ Bot başarıyla initialize edildi');
      console.log(`   Symbol: ${initResult.config.symbol}`);
      console.log(`   Leverage: ${initResult.config.leverage}x`);
      console.log(`   Testnet: ${initResult.config.testnet}`);
    } else {
      console.error('❌ Initialization başarısız:', initResult.error);

      // Bot zaten initialize ise devam et
      if (initResult.error.includes('already initialized')) {
        console.log('ℹ️  Bot zaten initialize edilmiş, devam ediliyor...');
      } else {
        return; // Başka bir hata varsa dur
      }
    }
  } catch (error) {
    console.error('❌ Initialization hatası:', error.message);
    return;
  }

  await sleep(1000);

  // Test 3: Get current metrics
  console.log('\n📊 Test 3: Bot metrics...');
  try {
    const metricsResponse = await fetch(`${API_URL}/api/monitoring/live`);
    const metricsResult = await metricsResponse.json();

    if (metricsResult.success) {
      console.log('✅ Metrics başarıyla alındı');
      console.log(`   Bot Status: ${metricsResult.data.bot.status}`);
      console.log(`   Running: ${metricsResult.data.bot.isRunning}`);
      console.log(`   Total Trades: ${metricsResult.data.performance.totalTrades}`);
      console.log(`   P&L: ${metricsResult.data.performance.totalPnL} USDT`);
      console.log(`   Compliance: ${metricsResult.data.compliance.status}`);
    } else {
      console.error('❌ Metrics başarısız:', metricsResult.error);
    }
  } catch (error) {
    console.error('❌ Metrics hatası:', error.message);
  }

  await sleep(1000);

  // Test 4: Start bot
  console.log('\n🚀 Test 4: Bot start...');
  try {
    const startResponse = await fetch(`${API_URL}/api/monitoring/live`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' }),
    });

    const startResult = await startResponse.json();

    if (startResult.success) {
      console.log('✅ Bot başarıyla başlatıldı');
      console.log('   ⚠️  Telegram/Discord\'da "Bot Started" alerti göreceksin');
    } else {
      console.error('❌ Bot start başarısız:', startResult.error);
    }
  } catch (error) {
    console.error('❌ Bot start hatası:', error.message);
  }

  await sleep(2000);

  // Test 5: Get running metrics
  console.log('\n📈 Test 5: Running bot metrics...');
  try {
    const runningMetricsResponse = await fetch(`${API_URL}/api/monitoring/live`);
    const runningMetricsResult = await runningMetricsResponse.json();

    if (runningMetricsResult.success) {
      console.log('✅ Running metrics alındı');
      console.log(`   Bot Status: ${runningMetricsResult.data.bot.status}`);
      console.log(`   Running: ${runningMetricsResult.data.bot.isRunning}`);
      console.log(`   Uptime: ${runningMetricsResult.data.bot.uptime}s`);
    } else {
      console.error('❌ Running metrics başarısız:', runningMetricsResult.error);
    }
  } catch (error) {
    console.error('❌ Running metrics hatası:', error.message);
  }

  await sleep(2000);

  // Test 6: Stop bot
  console.log('\n🛑 Test 6: Bot stop...');
  try {
    const stopResponse = await fetch(`${API_URL}/api/monitoring/live`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'stop' }),
    });

    const stopResult = await stopResponse.json();

    if (stopResult.success) {
      console.log('✅ Bot başarıyla durduruldu');
      console.log('   ⚠️  Telegram/Discord\'da "Bot Stopped" alerti göreceksin');
    } else {
      console.error('❌ Bot stop başarısız:', stopResult.error);
    }
  } catch (error) {
    console.error('❌ Bot stop hatası:', error.message);
  }

  await sleep(2000);

  // Test 7: Emergency stop
  console.log('\n🚨 Test 7: Emergency stop (CRITICAL alert)...');
  try {
    // Önce bot'u start et
    await fetch(`${API_URL}/api/monitoring/live`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'start' }),
    });

    await sleep(1000);

    // Emergency stop
    const emergencyResponse = await fetch(`${API_URL}/api/monitoring/live`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'emergency_stop' }),
    });

    const emergencyResult = await emergencyResponse.json();

    if (emergencyResult.success) {
      console.log('✅ Emergency stop başarılı');
      console.log('   🚨 CRITICAL ALERT:');
      console.log('      - Email (eğer configure edilmişse)');
      console.log('      - SMS (eğer configure edilmişse)');
      console.log('      - Telegram');
      console.log('      - Discord (eğer configure edilmişse)');
      console.log('      - Azure Event Hub');
    } else {
      console.error('❌ Emergency stop başarısız:', emergencyResult.error);
    }
  } catch (error) {
    console.error('❌ Emergency stop hatası:', error.message);
  }

  console.log('\n✅ TEST TAMAMLANDI!\n');
  console.log('📱 Alert Kanalları:');
  console.log('   - Telegram: 3 mesaj (Start, Stop, Emergency)');
  console.log('   - Discord: 3 mesaj (eğer configure edilmişse)');
  console.log('   - Console: Tüm log\'lar');
  console.log('\n🌐 Live Monitor: http://localhost:3000/live-monitor');
  console.log('');
}

// Check if server is running
async function checkServer() {
  try {
    const response = await fetch(`${API_URL}/api/monitoring/live`);
    return response.ok;
  } catch {
    return false;
  }
}

async function main() {
  const serverRunning = await checkServer();

  if (!serverRunning) {
    console.error('❌ HATA: Dev server çalışmıyor!');
    console.log('\nÖnce dev server\'ı başlat:');
    console.log('  cd ~/Desktop/borsa');
    console.log('  npm run dev\n');
    process.exit(1);
  }

  await testBotIntegration();
}

main().catch(console.error);
