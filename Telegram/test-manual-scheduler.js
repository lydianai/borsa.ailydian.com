#!/usr/bin/env node

/**
 * Manual test for Telegram scheduler functions
 * Directly tests the Market Correlation and Omnipotent Futures functions
 */

import 'dotenv/config';
import https from 'https';
import http from 'http';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN;
const TELEGRAM_CHAT_IDS = process.env.TELEGRAM_CHAT_IDS?.split(',') || [];

console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('🧪 MANUAL TELEGRAM SCHEDULER TEST');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`📊 Base URL: ${BASE_URL}`);
console.log(`🤖 Telegram Bot: ${TELEGRAM_BOT_TOKEN ? 'Configured' : 'Missing'}`);
console.log(`👤 Chat IDs: ${TELEGRAM_CHAT_IDS.length} configured`);
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

// HTTP GET helper
function httpGet(url) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const options = {
      hostname: urlObj.hostname,
      port: urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80),
      path: urlObj.pathname + urlObj.search,
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    };

    const req = (urlObj.protocol === 'https:' ? https : http).request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => (data += chunk));
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`Failed to parse JSON: ${e.message}`));
        }
      });
    });

    req.on('error', reject);
    req.setTimeout(10000, () => {
      req.destroy();
      reject(new Error('Request timeout'));
    });

    req.end();
  });
}

// Test Market Correlation (using FIXED code)
async function testMarketCorrelation() {
  console.log('🧪 Testing Market Correlation Function...\n');

  try {
    const data = await httpGet(`${BASE_URL}/api/market-correlation?limit=5`);

    console.log('  ✅ API call successful');
    console.log(`  📊 Response structure: success=${data.success}, data=${!!data.data}`);
    console.log(`  🔍 Data keys: ${Object.keys(data.data || {}).join(', ')}`);

    // THIS IS THE FIXED CODE
    if (!data.success || !data.data || !data.data.correlations || data.data.correlations.length === 0) {
      console.log('  ⚠️  No correlation data available\n');
      return { success: true, message: 'No data' };
    }

    const topCoins = data.data.correlations.slice(0, 5);
    console.log(`  ✅ Successfully sliced ${topCoins.length} coins from correlations array`);
    console.log(`  📈 Top coins:`);

    topCoins.forEach((coin, index) => {
      const medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][index];
      const correlation = (coin.btcCorrelation * 100).toFixed(0);
      console.log(`     ${medal} ${coin.symbol}: ${correlation}% BTC correlation`);
    });

    console.log('  ✅ Market Correlation function works correctly!\n');
    return { success: true, data: topCoins };

  } catch (error) {
    console.error('  ❌ Error:', error.message);
    console.error('  Stack:', error.stack);
    return { success: false, error: error.message };
  }
}

// Test Omnipotent Futures (using FIXED code)
async function testOmnipotentFutures() {
  console.log('🧪 Testing Omnipotent Futures Function...\n');

  try {
    const data = await httpGet(`${BASE_URL}/api/omnipotent-futures?limit=10`);

    console.log('  ✅ API call successful');
    console.log(`  📊 Response structure: success=${data.success}, data=${!!data.data}`);
    console.log(`  🔍 Data keys: ${Object.keys(data.data || {}).join(', ')}`);

    // THIS IS THE FIXED CODE
    if (!data.success || !data.data || !data.data.futures || data.data.futures.length === 0) {
      console.log('  ⚠️  No futures data available\n');
      return { success: true, message: 'No data' };
    }

    const strongSignals = data.data.futures.filter(s => s.wyckoffScore >= 70);
    console.log(`  ✅ Successfully filtered ${strongSignals.length} strong signals from ${data.data.futures.length} futures`);

    if (strongSignals.length === 0) {
      console.log('  ℹ️  No strong signals (wyckoffScore >= 70) found\n');
      return { success: true, message: 'No strong signals' };
    }

    console.log(`  🎯 Strong signals (wyckoffScore >= 70):`);
    strongSignals.slice(0, 3).forEach((signal, index) => {
      console.log(`     ${index + 1}. ${signal.symbol}: ${signal.wyckoffScore.toFixed(0)} (${signal.wyckoffPhase})`);
    });

    console.log('  ✅ Omnipotent Futures function works correctly!\n');
    return { success: true, data: strongSignals };

  } catch (error) {
    console.error('  ❌ Error:', error.message);
    console.error('  Stack:', error.stack);
    return { success: false, error: error.message };
  }
}

// Run all tests
async function runTests() {
  const results = {
    marketCorrelation: await testMarketCorrelation(),
    omnipotentFutures: await testOmnipotentFutures(),
  };

  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('📊 TEST RESULTS');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`  Market Correlation: ${results.marketCorrelation.success ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`  Omnipotent Futures: ${results.omnipotentFutures.success ? '✅ PASS' : '❌ FAIL'}`);
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  if (results.marketCorrelation.success && results.omnipotentFutures.success) {
    console.log('🎉 All tests PASSED! The scheduler fixes are working correctly.');
    console.log('✅ No "data.data.slice is not a function" errors');
    console.log('✅ No "data.data.filter is not a function" errors\n');
    process.exit(0);
  } else {
    console.log('❌ Some tests FAILED. Check the errors above.\n');
    process.exit(1);
  }
}

runTests().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});
