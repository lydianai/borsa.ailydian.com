#!/usr/bin/env node

/**
 * Manuel test for Top Buy Signals function
 */

const path = require('path');
require('dotenv').config({
  path: path.join(__dirname, '../.env.local')
});

const http = require('http');
const https = require('https');

const BASE_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

// HTTP GET helper
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
          resolve({ error: 'Parse error' });
        }
      });
    }).on('error', reject);
  });
}

async function testTopBuySignals() {
  console.log('🧪 Testing Top Buy Signals Function...\n');

  try {
    // Fetch all signal sources
    console.log('📡 Fetching signals from all sources...');
    const [aiSignals, quantumSignals, conservativeSignals, breakoutSignals] = await Promise.all([
      httpGet(`${BASE_URL}/api/ai-signals?limit=20`),
      httpGet(`${BASE_URL}/api/quantum-signals?limit=20`),
      httpGet(`${BASE_URL}/api/conservative-signals?limit=20`),
      httpGet(`${BASE_URL}/api/breakout-retest?limit=20`),
    ]);

    console.log('  ✅ AI Signals:', aiSignals.data?.signals?.length || 0);
    console.log('  ✅ Quantum Signals:', quantumSignals.data?.signals?.length || 0);
    console.log('  ✅ Conservative Signals:', conservativeSignals.data?.signals?.length || 0);
    console.log('  ✅ Breakout Signals:', breakoutSignals.data?.signals?.length || 0);

    // Collect BUY signals
    const allSignals = [];

    if (aiSignals.data?.signals) {
      aiSignals.data.signals.forEach(s => {
        if (s.type === 'BUY' && s.confidence >= 70) {
          allSignals.push({
            symbol: s.symbol,
            confidence: s.confidence,
            strategy: 'AI',
            price: s.price
          });
        }
      });
    }

    if (quantumSignals.data?.signals) {
      quantumSignals.data.signals.forEach(s => {
        if (s.type === 'BUY' && s.confidence >= 70) {
          allSignals.push({
            symbol: s.symbol,
            confidence: s.confidence,
            strategy: 'Quantum',
            price: s.price
          });
        }
      });
    }

    if (conservativeSignals.data?.signals) {
      conservativeSignals.data.signals.forEach(s => {
        if (s.type === 'BUY' && s.confidence >= 70) {
          allSignals.push({
            symbol: s.symbol,
            confidence: s.confidence,
            strategy: 'Conservative',
            price: s.price
          });
        }
      });
    }

    if (breakoutSignals.data?.signals) {
      breakoutSignals.data.signals.forEach(s => {
        if (s.type === 'BUY' && s.confidence >= 70) {
          allSignals.push({
            symbol: s.symbol,
            confidence: s.confidence,
            strategy: 'Breakout',
            price: s.price
          });
        }
      });
    }

    console.log(`\n📊 Total BUY signals collected: ${allSignals.length}`);

    // Sort by confidence
    allSignals.sort((a, b) => b.confidence - a.confidence);

    // Select unique coins
    const uniqueSignals = [];
    const seenSymbols = new Set();

    for (const signal of allSignals) {
      if (!seenSymbols.has(signal.symbol) && uniqueSignals.length < 7) {
        uniqueSignals.push(signal);
        seenSymbols.add(signal.symbol);
      }
    }

    console.log(`\n🎯 Top ${uniqueSignals.length} unique BUY signals:\n`);

    uniqueSignals.forEach((signal, index) => {
      const medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣', '7️⃣'][index];
      console.log(`${medal} ${signal.symbol}`);
      console.log(`   Strategy: ${signal.strategy}`);
      console.log(`   Confidence: ${signal.confidence}%`);
      console.log(`   Price: $${signal.price.toFixed(signal.price < 1 ? 6 : 2)}`);
      console.log('');
    });

    console.log('✅ Test completed successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

testTopBuySignals();
