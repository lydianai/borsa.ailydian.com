#!/usr/bin/env node
import axios from 'axios';

const BASE_URL = process.env.BASE_URL || 'http://localhost:3100';

interface TestResult {
  name: string;
  passed: boolean;
  duration: number;
  error?: string;
}

const results: TestResult[] = [];

async function test(name: string, fn: () => Promise<void>): Promise<void> {
  const startTime = Date.now();
  try {
    await fn();
    results.push({ name, passed: true, duration: Date.now() - startTime });
    console.log(`✅ ${name}`);
  } catch (error) {
    results.push({
      name,
      passed: false,
      duration: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error)
    });
    console.error(`❌ ${name}:`, error);
  }
}

async function runTests() {
  console.log('='.repeat(60));
  console.log('🧪 UNIFIED ROBOT ORCHESTRATOR - SMOKE TESTS');
  console.log('='.repeat(60));
  console.log('');

  await test('Orchestrator Control - Start', async () => {
    const response = await axios.post(`${BASE_URL}/api/orchestrator/control`, {
      action: 'start'
    }, { timeout: 30000 });

    if (!response.data.success) {
      throw new Error('Failed to start orchestrator');
    }

    await new Promise(resolve => setTimeout(resolve, 5000));
  });

  await test('Orchestrator Status', async () => {
    const response = await axios.get(`${BASE_URL}/api/orchestrator/status`);

    if (!response.data.success) {
      throw new Error('Status check failed');
    }

    const { orchestrator } = response.data;
    if (!orchestrator.isRunning) {
      throw new Error('Orchestrator is not running');
    }

    console.log(`   Total Bots: ${orchestrator.totalBots}`);
    console.log(`   Active Bots: ${orchestrator.activeBots}`);
  });

  await test('List Bots', async () => {
    const response = await axios.get(`${BASE_URL}/api/orchestrator/bots`);

    if (!response.data.success) {
      throw new Error('Failed to list bots');
    }

    if (response.data.bots.length === 0) {
      throw new Error('No bots registered');
    }

    console.log(`   Registered Bots: ${response.data.bots.length}`);
  });

  await test('Health Check', async () => {
    const response = await axios.post(`${BASE_URL}/api/orchestrator/health-check`);

    if (!response.data.success) {
      throw new Error('Health check failed');
    }

    const { summary } = response.data;
    console.log(`   Healthy: ${summary.healthy}/${summary.total}`);

    if (summary.healthy === 0) {
      throw new Error('No healthy bots');
    }
  });

  await test('Generate Single Signal - BTC/USDT', async () => {
    const response = await axios.post(`${BASE_URL}/api/orchestrator/signal`, {
      symbol: 'BTC/USDT'
    }, { timeout: 30000 });

    if (!response.data.success) {
      throw new Error('Signal generation failed');
    }

    const { consensus } = response.data;
    console.log(`   Action: ${consensus.action}`);
    console.log(`   Confidence: ${consensus.confidence.toFixed(2)}%`);
    console.log(`   Quality: ${consensus.quality}`);
    console.log(`   Bot Signals: ${consensus.botSignals.length}`);

    if (consensus.botSignals.length === 0) {
      throw new Error('No bot signals generated');
    }
  });

  await test('Generate Batch Signals', async () => {
    const response = await axios.post(`${BASE_URL}/api/orchestrator/signals/batch`, {
      symbols: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    }, { timeout: 60000 });

    if (!response.data.success) {
      throw new Error('Batch signal generation failed');
    }

    console.log(`   Generated: ${response.data.total}/${response.data.requested}`);

    if (response.data.total === 0) {
      throw new Error('No signals generated');
    }
  });

  await test('Performance Metrics', async () => {
    const response = await axios.get(`${BASE_URL}/api/orchestrator/metrics`);

    if (!response.data.success) {
      throw new Error('Metrics retrieval failed');
    }

    const { report } = response.data;
    console.log(`   Total Operations: ${report.totalOperations}`);
    console.log(`   Unique Operations: ${report.uniqueOperations}`);
    console.log(`   Success Rate: ${report.overallSuccessRate.toFixed(2)}%`);
  });

  console.log('');
  console.log('='.repeat(60));
  console.log('📊 TEST SUMMARY');
  console.log('='.repeat(60));

  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed).length;
  const total = results.length;
  const successRate = (passed / total) * 100;

  console.log(`Total Tests: ${total}`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Success Rate: ${successRate.toFixed(2)}%`);

  if (failed > 0) {
    console.log('');
    console.log('Failed Tests:');
    results.filter(r => !r.passed).forEach(r => {
      console.log(`  ❌ ${r.name}: ${r.error}`);
    });
  }

  console.log('='.repeat(60));

  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch(error => {
  console.error('Test suite failed:', error);
  process.exit(1);
});
