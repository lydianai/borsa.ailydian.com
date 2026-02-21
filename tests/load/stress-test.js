/**
 * k6 Stress Testing - Find Breaking Point
 *
 * Run with: k6 run tests/load/stress-test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Ramp up to 50 users
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '2m', target: 150 },  // Ramp up to 150 users
    { duration: '1m', target: 200 },  // Spike to 200 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<1000'], // 95% under 1s during stress
    errors: ['rate<0.2'],               // Up to 20% errors acceptable in stress test
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';

export default function () {
  const endpoints = [
    '/api/binance/futures',
    '/api/signals?limit=10',
    '/api/ai-signals?limit=10',
    '/api/conservative-signals',
    '/api/quantum-signals',
  ];

  // Random endpoint selection
  const endpoint = endpoints[Math.floor(Math.random() * endpoints.length)];
  const res = http.get(`${BASE_URL}${endpoint}`);

  check(res, {
    'status is 200 or 429': (r) => r.status === 200 || r.status === 429,
  });

  errorRate.add(res.status !== 200 && res.status !== 429);
  sleep(0.5);
}
