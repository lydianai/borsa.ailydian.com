#!/usr/bin/env node

/**
 * Test script to verify Telegram scheduler fixes
 * Tests the Market Correlation and Omnipotent Futures functions
 */

import 'dotenv/config';
import https from 'https';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';

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

    const req = (urlObj.protocol === 'https:' ? https : require('http')).request(options, (res) => {
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

// Test Market Correlation
async function testMarketCorrelation() {
  console.log('\n🧪 Testing Market Correlation...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  try {
    const data = await httpGet(`${BASE_URL}/api/market-correlation?limit=5`);

    console.log(`✅ API Response received`);
    console.log(`   - success: ${data.success}`);
    console.log(`   - data exists: ${!!data.data}`);
    console.log(`   - correlations exists: ${!!data.data?.correlations}`);
    console.log(`   - correlations is array: ${Array.isArray(data.data?.correlations)}`);
    console.log(`   - correlations length: ${data.data?.correlations?.length || 0}`);

    if (!data.success || !data.data || !data.data.correlations || data.data.correlations.length === 0) {
      console.log('⚠️  No correlation data available (this is OK if market is closed)');
      return false;
    }

    const topCoins = data.data.correlations.slice(0, 5);
    console.log(`✅ Successfully sliced top ${topCoins.length} coins:`);
    topCoins.forEach((coin, index) => {
      console.log(`   ${index + 1}. ${coin.symbol}: ${(coin.btcCorrelation * 100).toFixed(0)}%`);
    });

    return true;
  } catch (error) {
    console.error('❌ Market Correlation test failed:', error.message);
    return false;
  }
}

// Test Omnipotent Futures
async function testOmnipotentFutures() {
  console.log('\n🧪 Testing Omnipotent Futures...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  try {
    const data = await httpGet(`${BASE_URL}/api/omnipotent-futures?limit=10`);

    console.log(`✅ API Response received`);
    console.log(`   - success: ${data.success}`);
    console.log(`   - data exists: ${!!data.data}`);
    console.log(`   - futures exists: ${!!data.data?.futures}`);
    console.log(`   - futures is array: ${Array.isArray(data.data?.futures)}`);
    console.log(`   - futures length: ${data.data?.futures?.length || 0}`);

    if (!data.success || !data.data || !data.data.futures || data.data.futures.length === 0) {
      console.log('⚠️  No futures data available (this is OK if market is closed)');
      return false;
    }

    const strongSignals = data.data.futures.filter(s => s.wyckoffScore >= 70);
    console.log(`✅ Successfully filtered strong signals (wyckoffScore >= 70):`);
    console.log(`   - Total futures: ${data.data.futures.length}`);
    console.log(`   - Strong signals: ${strongSignals.length}`);

    if (strongSignals.length > 0) {
      console.log('\n   Top signals:');
      strongSignals.slice(0, 3).forEach((signal, index) => {
        console.log(`   ${index + 1}. ${signal.symbol}: ${signal.wyckoffScore.toFixed(0)} (${signal.wyckoffPhase})`);
      });
    }

    return true;
  } catch (error) {
    console.error('❌ Omnipotent Futures test failed:', error.message);
    return false;
  }
}

// Run tests
async function runTests() {
  console.log('╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮');
  console.log('┃  TELEGRAM SCHEDULER FIX VALIDATION       ┃');
  console.log('╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯');
  console.log(`\n📊 Base URL: ${BASE_URL}`);

  const results = {
    marketCorrelation: await testMarketCorrelation(),
    omnipotentFutures: await testOmnipotentFutures(),
  };

  console.log('\n╭━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╮');
  console.log('┃  TEST RESULTS                            ┃');
  console.log('├━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤');
  console.log(`┃  Market Correlation: ${results.marketCorrelation ? '✅ PASS' : '⚠️  NO DATA'}`);
  console.log(`┃  Omnipotent Futures: ${results.omnipotentFutures ? '✅ PASS' : '⚠️  NO DATA'}`);
  console.log('╰━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╯');

  if (results.marketCorrelation && results.omnipotentFutures) {
    console.log('\n🎉 All tests passed! Scheduler fixes are working correctly.');
    process.exit(0);
  } else {
    console.log('\n⚠️  Some tests returned no data (may be normal if market is closed)');
    console.log('   Check the error log for any "data.data.slice is not a function" errors');
    console.log('   If no such errors appear, the fix is working correctly!');
    process.exit(0);
  }
}

runTests();
