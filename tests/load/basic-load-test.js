/**
 * k6 Load Testing - Basic Scenario
 *
 * White-hat compliance: Load testing ensures system reliability
 * and identifies performance bottlenecks
 *
 * Run with: k6 run tests/load/basic-load-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests must complete below 500ms
    errors: ['rate<0.1'],              // Error rate must be below 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  // Test 1: Market Data API
  const marketDataRes = http.get(`${BASE_URL}/api/binance/futures`);
  check(marketDataRes, {
    'market data status is 200': (r) => r.status === 200,
    'market data has data': (r) => {
      const body = JSON.parse(r.body);
      return body.success === true && Array.isArray(body.data?.all);
    },
  });
  errorRate.add(marketDataRes.status !== 200);
  responseTime.add(marketDataRes.timings.duration);

  sleep(1);

  // Test 2: Signals API
  const signalsRes = http.get(`${BASE_URL}/api/signals?limit=10`);
  check(signalsRes, {
    'signals status is 200': (r) => r.status === 200,
    'signals has data': (r) => {
      const body = JSON.parse(r.body);
      return body.success === true;
    },
  });
  errorRate.add(signalsRes.status !== 200);
  responseTime.add(signalsRes.timings.duration);

  sleep(1);

  // Test 3: AI Signals API
  const aiSignalsRes = http.get(`${BASE_URL}/api/ai-signals?limit=10`);
  check(aiSignalsRes, {
    'ai signals status is 200': (r) => r.status === 200,
  });
  errorRate.add(aiSignalsRes.status !== 200);
  responseTime.add(aiSignalsRes.timings.duration);

  sleep(2);
}
