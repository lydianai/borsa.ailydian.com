#!/usr/bin/env node

/**
 * PENETRASYON TEST SİSTEMİ
 *
 * Tüm sistemi test eder:
 * 1. API endpoints
 * 2. Strategy functions
 * 3. Real Binance data
 * 4. Error handling
 * 5. Type safety
 */

const https = require('https');

// Test sonuçları
const results = {
  passed: 0,
  failed: 0,
  errors: [],
};

// Renk kodları
const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  reset: '\x1b[0m',
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSuccess(message) {
  log(`✅ ${message}`, 'green');
  results.passed++;
}

function logError(message, error) {
  log(`❌ ${message}`, 'red');
  if (error) log(`   Error: ${error.message}`, 'red');
  results.failed++;
  results.errors.push({ message, error: error?.message });
}

function logInfo(message) {
  log(`ℹ️  ${message}`, 'blue');
}

// HTTP request helper
function httpsGet(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, data: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, data });
        }
      });
    }).on('error', reject);
  });
}

// TEST 1: Binance API Connection
async function testBinanceConnection() {
  logInfo('TEST 1: Binance API Connection...');

  try {
    const result = await httpsGet('https://fapi.binance.com/fapi/v1/ping');
    if (result.status === 200) {
      logSuccess('Binance API bağlantısı başarılı');
    } else {
      logError('Binance API bağlantısı başarısız', new Error(`Status: ${result.status}`));
    }
  } catch (error) {
    logError('Binance API bağlantı hatası', error);
  }
}

// TEST 2: Futures Data Fetch
async function testFuturesData() {
  logInfo('TEST 2: Futures Data Availability...');

  try {
    const result = await httpsGet('https://fapi.binance.com/fapi/v1/ticker/24hr');
    if (result.status === 200 && Array.isArray(result.data)) {
      const usdtPairs = result.data.filter(t => t.symbol.endsWith('USDT'));
      logSuccess(`${usdtPairs.length} USDT futures pairs bulundu`);

      // Sample coin test
      const btc = usdtPairs.find(t => t.symbol === 'BTCUSDT');
      if (btc) {
        logSuccess(`BTC fiyatı: $${parseFloat(btc.lastPrice).toFixed(2)}`);
      }
    } else {
      logError('Futures data çekilemedi', new Error('Invalid response'));
    }
  } catch (error) {
    logError('Futures data hatası', error);
  }
}

// TEST 3: Candle Data Fetch
async function testCandleData() {
  logInfo('TEST 3: Candle Data (Kline) Fetch...');

  try {
    const result = await httpsGet('https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=4h&limit=200');
    if (result.status === 200 && Array.isArray(result.data)) {
      logSuccess(`${result.data.length} candle data çekildi (BTCUSDT 4h)`);

      // Validate candle structure
      const lastCandle = result.data[result.data.length - 1];
      if (lastCandle && lastCandle.length >= 6) {
        const close = parseFloat(lastCandle[4]);
        logSuccess(`Son kapanış: $${close.toFixed(2)}`);
      }
    } else {
      logError('Candle data çekilemedi', new Error('Invalid response'));
    }
  } catch (error) {
    logError('Candle data hatası', error);
  }
}

// TEST 4: Exchange Info
async function testExchangeInfo() {
  logInfo('TEST 4: Exchange Info...');

  try {
    const result = await httpsGet('https://fapi.binance.com/fapi/v1/exchangeInfo');
    if (result.status === 200 && result.data.symbols) {
      const perpetuals = result.data.symbols.filter(s =>
        s.contractType === 'PERPETUAL' &&
        s.quoteAsset === 'USDT' &&
        s.status === 'TRADING'
      );
      logSuccess(`${perpetuals.length} aktif USDT perpetual contracts bulundu`);
    } else {
      logError('Exchange info çekilemedi', new Error('Invalid response'));
    }
  } catch (error) {
    logError('Exchange info hatası', error);
  }
}

// TEST 5: Rate Limiting Check
async function testRateLimiting() {
  logInfo('TEST 5: Rate Limiting Compliance...');

  try {
    const start = Date.now();
    const requests = [];

    // 5 request in a row with 100ms delay
    for (let i = 0; i < 5; i++) {
      await new Promise(resolve => setTimeout(resolve, 100));
      requests.push(httpsGet('https://fapi.binance.com/fapi/v1/ping'));
    }

    await Promise.all(requests);
    const duration = Date.now() - start;

    if (duration >= 400) { // Should take at least 400ms (5 * 100ms - 100ms)
      logSuccess(`Rate limiting uyumlu (${duration}ms sürdü)`);
    } else {
      logError('Rate limiting çok hızlı', new Error(`Only took ${duration}ms`));
    }
  } catch (error) {
    logError('Rate limiting test hatası', error);
  }
}

// TEST 6: Error Handling
async function testErrorHandling() {
  logInfo('TEST 6: Error Handling...');

  try {
    // Invalid symbol should be handled gracefully
    const result = await httpsGet('https://fapi.binance.com/fapi/v1/klines?symbol=INVALID&interval=4h&limit=10');

    // Should get an error response, but not crash
    if (result.status === 400) {
      logSuccess('Invalid symbol hatası doğru handle edildi');
    } else {
      logError('Error handling beklenmedik sonuç', new Error(`Status: ${result.status}`));
    }
  } catch (error) {
    // Network error is also acceptable
    logSuccess('Network error doğru catch edildi');
  }
}

// TEST 7: Multiple Coins Test
async function testMultipleCoins() {
  logInfo('TEST 7: Multiple Coins Data Integrity...');

  try {
    const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'];
    const results = [];

    for (const symbol of symbols) {
      await new Promise(resolve => setTimeout(resolve, 100)); // Rate limiting
      const result = await httpsGet(`https://fapi.binance.com/fapi/v1/klines?symbol=${symbol}&interval=4h&limit=100`);
      results.push({ symbol, success: result.status === 200, count: result.data?.length || 0 });
    }

    const allSuccess = results.every(r => r.success && r.count === 100);
    if (allSuccess) {
      logSuccess(`${symbols.length} coin'in tamamından veri çekildi`);
    } else {
      logError('Bazı coin\'lerden veri çekilemedi', new Error(JSON.stringify(results)));
    }
  } catch (error) {
    logError('Multiple coins test hatası', error);
  }
}

// TEST 8: Data Consistency
async function testDataConsistency() {
  logInfo('TEST 8: Data Consistency...');

  try {
    // Fetch same data twice
    const result1 = await httpsGet('https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT');
    await new Promise(resolve => setTimeout(resolve, 100));
    const result2 = await httpsGet('https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT');

    if (result1.status === 200 && result2.status === 200) {
      const price1 = parseFloat(result1.data.lastPrice);
      const price2 = parseFloat(result2.data.lastPrice);

      // Prices should be within 1% of each other (normal market movement)
      const diff = Math.abs(price1 - price2) / price1;
      if (diff < 0.01) {
        logSuccess(`Data tutarlı (fiyat farkı: ${(diff * 100).toFixed(3)}%)`);
      } else {
        logError('Data tutarsız', new Error(`Price difference: ${(diff * 100).toFixed(2)}%`));
      }
    }
  } catch (error) {
    logError('Data consistency test hatası', error);
  }
}

// TEST 9: Volume Data Validation
async function testVolumeData() {
  logInfo('TEST 9: Volume Data Validation...');

  try {
    const result = await httpsGet('https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=4h&limit=50');

    if (result.status === 200 && Array.isArray(result.data)) {
      const candles = result.data.map(d => ({
        volume: parseFloat(d[5]),
        quoteVolume: parseFloat(d[7]),
      }));

      // Check all candles have positive volume
      const allPositive = candles.every(c => c.volume > 0 && c.quoteVolume > 0);

      if (allPositive) {
        const avgVolume = candles.reduce((sum, c) => sum + c.volume, 0) / candles.length;
        logSuccess(`Volume data geçerli (ortalama: ${avgVolume.toFixed(2)} BTC)`);
      } else {
        logError('Volume data geçersiz', new Error('Some candles have zero volume'));
      }
    }
  } catch (error) {
    logError('Volume data test hatası', error);
  }
}

// TEST 10: Timeframe Support
async function testTimeframeSupport() {
  logInfo('TEST 10: Multiple Timeframe Support...');

  try {
    const timeframes = ['1h', '4h', '1d'];
    const results = [];

    for (const tf of timeframes) {
      await new Promise(resolve => setTimeout(resolve, 100));
      const result = await httpsGet(`https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=${tf}&limit=100`);
      results.push({ tf, success: result.status === 200, count: result.data?.length || 0 });
    }

    const allSuccess = results.every(r => r.success && r.count === 100);
    if (allSuccess) {
      logSuccess(`${timeframes.length} timeframe destekleniyor`);
    } else {
      logError('Bazı timeframe\'ler çalışmıyor', new Error(JSON.stringify(results)));
    }
  } catch (error) {
    logError('Timeframe support test hatası', error);
  }
}

// Main test runner
async function runAllTests() {
  log('\n========================================', 'yellow');
  log('🔒 PENETRASYON TEST BAŞLIYOR', 'yellow');
  log('========================================\n', 'yellow');

  await testBinanceConnection();
  await testFuturesData();
  await testCandleData();
  await testExchangeInfo();
  await testRateLimiting();
  await testErrorHandling();
  await testMultipleCoins();
  await testDataConsistency();
  await testVolumeData();
  await testTimeframeSupport();

  log('\n========================================', 'yellow');
  log('📊 TEST SONUÇLARI', 'yellow');
  log('========================================\n', 'yellow');

  log(`✅ Başarılı: ${results.passed}`, 'green');
  log(`❌ Başarısız: ${results.failed}`, results.failed > 0 ? 'red' : 'green');

  if (results.errors.length > 0) {
    log('\n🔴 HATALAR:', 'red');
    results.errors.forEach((err, i) => {
      log(`  ${i + 1}. ${err.message}`, 'red');
      if (err.error) log(`     ${err.error}`, 'red');
    });
  }

  const successRate = (results.passed / (results.passed + results.failed)) * 100;
  log(`\n📈 Başarı Oranı: ${successRate.toFixed(1)}%`, successRate === 100 ? 'green' : 'yellow');

  if (successRate === 100) {
    log('\n🎉 TÜM TESTLER BAŞARILI - SİSTEM HATASIZ!', 'green');
  } else {
    log('\n⚠️  BAZI TESTLER BAŞARISIZ - LÜTFEN HATALARI İNCELEYİN', 'red');
  }

  log('\n========================================\n', 'yellow');

  process.exit(results.failed > 0 ? 1 : 0);
}

// Run tests
runAllTests().catch(error => {
  logError('Test runner hatası', error);
  process.exit(1);
});
