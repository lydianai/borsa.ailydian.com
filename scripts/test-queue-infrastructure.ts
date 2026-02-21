/**
 * QUEUE INFRASTRUCTURE TEST
 * Tests the complete queue flow: enqueue → worker → metrics
 *
 * Usage:
 * npx ts-node scripts/test-queue-infrastructure.ts
 */

import { createHmac } from 'crypto';

const QUEUE_DRIVER = process.env.QUEUE_DRIVER || 'memory';
const INTERNAL_SERVICE_TOKEN = process.env.INTERNAL_SERVICE_TOKEN || 'test-token-local';
const BASE_URL = process.env.TEST_BASE_URL || 'http://localhost:3000';

console.log('============================================================');
console.log('🧪 QUEUE INFRASTRUCTURE TEST');
console.log('============================================================\n');
console.log(`Driver: ${QUEUE_DRIVER}`);
console.log(`Base URL: ${BASE_URL}`);
console.log(`Token: ${INTERNAL_SERVICE_TOKEN.substring(0, 10)}****\n`);

/**
 * Test 1: Health Check
 */
async function testHealthCheck() {
  console.log('📋 Test 1: Health Check');
  console.log('─────────────────────────────────────────────────────────');

  try {
    const response = await fetch(`${BASE_URL}/api/health`);
    const data = await response.json();

    console.log(`✅ Status: ${response.status}`);
    console.log(`✅ Response:`, JSON.stringify(data, null, 2));
    console.log('');
    return true;
  } catch (error: any) {
    console.error(`❌ Health check failed:`, error.message);
    console.log('');
    return false;
  }
}

/**
 * Test 2: Queue Metrics (Before Enqueue)
 */
async function testMetricsBefore() {
  console.log('📊 Test 2: Queue Metrics (Before Enqueue)');
  console.log('─────────────────────────────────────────────────────────');

  try {
    const response = await fetch(`${BASE_URL}/api/queue/metrics`, {
      headers: {
        Authorization: `Bearer ${INTERNAL_SERVICE_TOKEN}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    console.log(`✅ Status: ${response.status}`);
    console.log(`✅ Health: ${data.health.healthy ? 'Healthy' : 'Unhealthy'}`);
    console.log(`✅ Driver: ${data.health.driver}`);
    console.log(`✅ Queue State:`, JSON.stringify(data.metrics.queue, null, 2));
    console.log('');
    return true;
  } catch (error: any) {
    console.error(`❌ Metrics check failed:`, error.message);
    console.log('');
    return false;
  }
}

/**
 * Test 3: Enqueue Job
 */
async function testEnqueueJob() {
  console.log('📥 Test 3: Enqueue Job');
  console.log('─────────────────────────────────────────────────────────');

  const requestBody = {
    requestId: `test-${Date.now()}`,
    requestedBy: 'test-script',
    scopes: ['scan:enqueue'],
    symbols: ['BTCUSDT', 'ETHUSDT'],
    strategies: ['ma-pullback', 'rsi-divergence'],
    priority: 5,
  };

  const payload = JSON.stringify(requestBody);

  // Create HMAC signature
  const signature = createHmac('sha256', INTERNAL_SERVICE_TOKEN)
    .update(payload)
    .digest('hex');

  try {
    const response = await fetch(`${BASE_URL}/api/queue/enqueue`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-signature': signature,
        'x-client-id': 'test-script',
      },
      body: payload,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    console.log(`✅ Status: ${response.status}`);
    console.log(`✅ Job ID: ${data.jobId}`);
    console.log(`✅ Request ID: ${data.requestId}`);
    console.log(`✅ Queued: ${data.queued.symbols} symbols, ${data.queued.strategies} strategies`);
    console.log('');
    return { success: true, jobId: data.jobId };
  } catch (error: any) {
    console.error(`❌ Enqueue failed:`, error.message);
    console.log('');
    return { success: false };
  }
}

/**
 * Test 4: Queue Metrics (After Enqueue)
 */
async function testMetricsAfter() {
  console.log('📊 Test 4: Queue Metrics (After Enqueue)');
  console.log('─────────────────────────────────────────────────────────');

  // Wait 2 seconds for job to be processed
  console.log('⏳ Waiting 2 seconds for job processing...');
  await new Promise((resolve) => setTimeout(resolve, 2000));

  try {
    const response = await fetch(`${BASE_URL}/api/queue/metrics`, {
      headers: {
        Authorization: `Bearer ${INTERNAL_SERVICE_TOKEN}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();

    console.log(`✅ Status: ${response.status}`);
    console.log(`✅ Queue State:`, JSON.stringify(data.metrics.queue, null, 2));
    console.log(`✅ Computed:`, JSON.stringify(data.metrics.computed, null, 2));
    console.log('');
    return true;
  } catch (error: any) {
    console.error(`❌ Metrics check failed:`, error.message);
    console.log('');
    return false;
  }
}

/**
 * Test 5: Rate Limiting
 */
async function testRateLimiting() {
  console.log('🚦 Test 5: Rate Limiting');
  console.log('─────────────────────────────────────────────────────────');

  const requestBody = {
    requestId: `rate-limit-test-${Date.now()}`,
    requestedBy: 'rate-limit-test',
    scopes: ['scan:enqueue'],
    symbols: ['BTCUSDT'],
    strategies: ['ma-pullback'],
    priority: 5,
  };

  const payload = JSON.stringify(requestBody);
  const signature = createHmac('sha256', INTERNAL_SERVICE_TOKEN)
    .update(payload)
    .digest('hex');

  // Send 15 requests rapidly (default limit is 10/min)
  let successCount = 0;
  let rateLimitedCount = 0;

  console.log('📤 Sending 15 rapid requests...');

  for (let i = 0; i < 15; i++) {
    try {
      const response = await fetch(`${BASE_URL}/api/queue/enqueue`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-signature': signature,
          'x-client-id': 'rate-limit-test',
        },
        body: payload,
      });

      if (response.status === 429) {
        rateLimitedCount++;
      } else if (response.ok) {
        successCount++;
      }
    } catch (error) {
      // Ignore errors
    }
  }

  console.log(`✅ Successful: ${successCount}`);
  console.log(`✅ Rate Limited: ${rateLimitedCount}`);
  console.log(`✅ Rate limiting ${rateLimitedCount > 0 ? 'WORKING' : 'NOT WORKING (check config)'}`);
  console.log('');

  return rateLimitedCount > 0;
}

/**
 * Main Test Runner
 */
async function runTests() {
  const results = {
    healthCheck: false,
    metricsBefore: false,
    enqueueJob: false,
    metricsAfter: false,
    rateLimiting: false,
  };

  try {
    results.healthCheck = await testHealthCheck();
    results.metricsBefore = await testMetricsBefore();

    const enqueueResult = await testEnqueueJob();
    results.enqueueJob = enqueueResult.success;

    results.metricsAfter = await testMetricsAfter();
    results.rateLimiting = await testRateLimiting();
  } catch (error: any) {
    console.error('❌ Test suite error:', error.message);
  }

  // Summary
  console.log('============================================================');
  console.log('📝 TEST SUMMARY');
  console.log('============================================================\n');

  const tests = [
    { name: 'Health Check', result: results.healthCheck },
    { name: 'Metrics Before', result: results.metricsBefore },
    { name: 'Enqueue Job', result: results.enqueueJob },
    { name: 'Metrics After', result: results.metricsAfter },
    { name: 'Rate Limiting', result: results.rateLimiting },
  ];

  tests.forEach((test) => {
    console.log(`${test.result ? '✅' : '❌'} ${test.name}`);
  });

  const passedCount = tests.filter((t) => t.result).length;
  const totalCount = tests.length;

  console.log('');
  console.log(`Total: ${passedCount}/${totalCount} tests passed`);
  console.log('');

  if (passedCount === totalCount) {
    console.log('🎉 ALL TESTS PASSED! Queue infrastructure is working correctly.');
  } else {
    console.log('⚠️  SOME TESTS FAILED. Check the logs above for details.');
  }

  console.log('============================================================\n');

  process.exit(passedCount === totalCount ? 0 : 1);
}

// Run tests
runTests();
