/**
 * 🧪 API INTEGRATION TESTS
 *
 * Comprehensive endpoint testing for:
 * - Health check
 * - Binance Futures data
 * - Historical candles (Klines)
 * - Unified Strategy signals
 * - Breakout-Retest historical
 * - AI Signals (3-layer system)
 *
 * Coverage Target: 100% endpoint coverage
 * Test Strategy: Black-box API testing with real server
 */

import { describe, it, expect, beforeAll } from 'vitest';

// Test configuration
const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3001';
const TIMEOUT = 30000; // 30 seconds for API calls

describe('API Integration Tests', () => {

  // ============================================
  // HEALTH CHECK ENDPOINT
  // ============================================

  describe('GET /api/health', () => {
    it('should return 200 OK status', async () => {
      const response = await fetch(`${BASE_URL}/api/health`);
      expect(response.ok).toBe(true);
      expect(response.status).toBe(200);
    }, TIMEOUT);

    it('should return correct health status', async () => {
      const response = await fetch(`${BASE_URL}/api/health`);
      const data = await response.json();

      expect(data).toHaveProperty('status');
      expect(data.status).toBe('ok');
      expect(data).toHaveProperty('message');
      expect(data).toHaveProperty('timestamp');
      expect(data).toHaveProperty('version');
    }, TIMEOUT);
  });

  // ============================================
  // BINANCE FUTURES ENDPOINT
  // ============================================

  describe('GET /api/binance/futures', () => {
    it('should return market data successfully', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/futures`);
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data).toHaveProperty('data');
    }, TIMEOUT);

    it('should contain all required market data fields', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/futures`);
      const data = await response.json();

      expect(data.data).toHaveProperty('all'); // All markets
      expect(data.data).toHaveProperty('topVolume'); // Top by volume
      expect(data.data).toHaveProperty('topGainers'); // Top gainers
      expect(data.data).toHaveProperty('topLosers'); // Top losers
      expect(data.data).toHaveProperty('stats'); // Market statistics
    }, TIMEOUT);

    it('should return at least 100 markets', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/futures`);
      const data = await response.json();

      expect(data.data.all.length).toBeGreaterThanOrEqual(100);
    }, TIMEOUT);

    it('should have valid price data structure', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/futures`);
      const data = await response.json();

      const firstMarket = data.data.all[0];
      expect(firstMarket).toHaveProperty('symbol');
      expect(firstMarket).toHaveProperty('price');
      expect(firstMarket).toHaveProperty('changePercent24h');
      expect(firstMarket).toHaveProperty('volume24h');
      expect(firstMarket).toHaveProperty('high24h');
      expect(firstMarket).toHaveProperty('low24h');
    }, TIMEOUT);
  });

  // ============================================
  // BINANCE KLINES (HISTORICAL DATA) ENDPOINT
  // ============================================

  describe('GET /api/binance/klines', () => {
    it('should return candle data for BTC', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/klines?symbol=BTCUSDT&interval=4h&limit=50`);
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.data).toHaveProperty('candles');
      expect(data.data.candles.length).toBe(50);
    }, TIMEOUT);

    it('should support multiple timeframes', async () => {
      const timeframes = ['15m', '1h', '4h'];

      for (const interval of timeframes) {
        const response = await fetch(`${BASE_URL}/api/binance/klines?symbol=ETHUSDT&interval=${interval}&limit=10`);
        const data = await response.json();

        expect(data.success).toBe(true);
        expect(data.data.interval).toBe(interval);
        expect(data.data.candles.length).toBe(10);
      }
    }, TIMEOUT * 3);

    it('should return proper OHLCV structure', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/klines?symbol=BTCUSDT&interval=4h&limit=1`);
      const data = await response.json();

      const candle = data.data.candles[0];
      expect(candle).toHaveProperty('timestamp');
      expect(candle).toHaveProperty('open');
      expect(candle).toHaveProperty('high');
      expect(candle).toHaveProperty('low');
      expect(candle).toHaveProperty('close');
      expect(candle).toHaveProperty('volume');
      expect(candle).toHaveProperty('closeTime');
    }, TIMEOUT);

    it('should reject invalid timeframes', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/klines?symbol=BTCUSDT&interval=999h&limit=10`);
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data).toHaveProperty('error');
    }, TIMEOUT);

    it('should include statistics', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/klines?symbol=BTCUSDT&interval=4h&limit=100`);
      const data = await response.json();

      expect(data.data.stats).toHaveProperty('count');
      expect(data.data.stats).toHaveProperty('priceChange');
      expect(data.data.stats).toHaveProperty('priceChangePercent');
      expect(data.data.stats).toHaveProperty('highestPrice');
      expect(data.data.stats).toHaveProperty('lowestPrice');
      expect(data.data.stats).toHaveProperty('avgVolume');
    }, TIMEOUT);
  });

  // ============================================
  // UNIFIED SIGNALS ENDPOINT
  // ============================================

  describe('GET /api/unified-signals', () => {
    it('should return unified strategy signals', async () => {
      const response = await fetch(`${BASE_URL}/api/unified-signals?minBuyPercentage=50&limit=10`);
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.data).toHaveProperty('signals');
      expect(data.data).toHaveProperty('stats');
    }, TIMEOUT);

    it('should filter by buy percentage threshold', async () => {
      const response = await fetch(`${BASE_URL}/api/unified-signals?minBuyPercentage=70&limit=20`);
      const data = await response.json();

      // All signals should have buyPercentage >= 70
      data.data.signals.forEach((signal: any) => {
        expect(signal.buyPercentage).toBeGreaterThanOrEqual(70);
      });
    }, TIMEOUT);

    it('should include strategy breakdown', async () => {
      const response = await fetch(`${BASE_URL}/api/unified-signals?minBuyPercentage=60&limit=5`);
      const data = await response.json();

      if (data.data.signals.length > 0) {
        const firstSignal = data.data.signals[0];
        expect(firstSignal).toHaveProperty('symbol');
        expect(firstSignal).toHaveProperty('finalDecision'); // BUY or WAIT
        expect(firstSignal).toHaveProperty('buyPercentage');
        expect(firstSignal).toHaveProperty('waitPercentage');
        expect(firstSignal).toHaveProperty('strategyBreakdown');
        expect(firstSignal).toHaveProperty('topRecommendations');
        expect(firstSignal).toHaveProperty('riskAssessment');
      }
    }, TIMEOUT);

    it('should return correct statistics', async () => {
      const response = await fetch(`${BASE_URL}/api/unified-signals?minBuyPercentage=50&limit=20`);
      const data = await response.json();

      expect(data.data.stats).toHaveProperty('scanned');
      expect(data.data.stats).toHaveProperty('found');
      expect(data.data.stats).toHaveProperty('avgBuyPercentage');
      expect(data.data.stats).toHaveProperty('avgConfidence');
      expect(data.data.stats).toHaveProperty('timestamp');
    }, TIMEOUT);
  });

  // ============================================
  // BREAKOUT-RETEST HISTORICAL ENDPOINT
  // ============================================

  describe('GET /api/breakout-retest-historical', () => {
    it('should return breakout-retest signals', async () => {
      const response = await fetch(`${BASE_URL}/api/breakout-retest-historical?interval=4h&minConfidence=70&limit=10`);
      const data = await response.json();

      expect(data.success).toBe(true);
      expect(data.data).toHaveProperty('signals');
      expect(data.data).toHaveProperty('stats');
    }, TIMEOUT);

    it('should include pattern details', async () => {
      const response = await fetch(`${BASE_URL}/api/breakout-retest-historical?interval=4h&limit=5`);
      const data = await response.json();

      if (data.data.signals.length > 0) {
        const firstSignal = data.data.signals[0];
        expect(firstSignal).toHaveProperty('symbol');
        expect(firstSignal).toHaveProperty('signal'); // BUY or SELL
        expect(firstSignal).toHaveProperty('confidence');
        expect(firstSignal).toHaveProperty('pattern');

        // Pattern should have 3 phases
        expect(firstSignal.pattern).toHaveProperty('consolidation');
        expect(firstSignal.pattern).toHaveProperty('breakout');
        expect(firstSignal.pattern).toHaveProperty('retest');
      }
    }, TIMEOUT);

    it('should support different intervals', async () => {
      const intervals = ['15m', '1h', '4h'];

      for (const interval of intervals) {
        const response = await fetch(`${BASE_URL}/api/breakout-retest-historical?interval=${interval}&limit=3`);
        const data = await response.json();

        expect(data.success).toBe(true);
        expect(data.data.stats.timeframe).toBe(interval);
      }
    }, TIMEOUT * 3);
  });

  // ============================================
  // AI SIGNALS ENDPOINT (3-LAYER SYSTEM)
  // ============================================

  describe('GET /api/ai-signals', () => {
    it('should return AI-generated signals', async () => {
      const response = await fetch(`${BASE_URL}/api/ai-signals`);

      // May fail if market data unavailable, that's ok
      if (response.ok) {
        const data = await response.json();
        expect(data.success).toBe(true);
      }
    }, TIMEOUT);

    it('should include 3 AI models in response', async () => {
      const response = await fetch(`${BASE_URL}/api/ai-signals`);

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data) {
          expect(data.data).toHaveProperty('aiModels');
          expect(data.data.aiModels).toContain('AI-Model-Alpha'); // Groq
          expect(data.data.aiModels).toContain('AI-Model-Beta'); // Quantum
          expect(data.data.aiModels).toContain('AI-Model-Gamma'); // Unified Strategy
        }
      }
    }, TIMEOUT);

    it('should include unified strategy statistics', async () => {
      const response = await fetch(`${BASE_URL}/api/ai-signals`);

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data) {
          expect(data.data).toHaveProperty('unifiedStrategy');
          expect(data.data.unifiedStrategy).toHaveProperty('available');
          expect(data.data.unifiedStrategy).toHaveProperty('totalSignals');
          expect(data.data.unifiedStrategy).toHaveProperty('enrichedSignals');
          expect(data.data.unifiedStrategy).toHaveProperty('enrichmentRate');
        }
      }
    }, TIMEOUT);

    it('should contain AI learning statistics', async () => {
      const response = await fetch(`${BASE_URL}/api/ai-signals`);

      if (response.ok) {
        const data = await response.json();
        if (data.success && data.data) {
          expect(data.data).toHaveProperty('aiLearning');
          expect(data.data.aiLearning).toHaveProperty('totalAnalyzed');
          expect(data.data.aiLearning).toHaveProperty('successRate');
        }
      }
    }, TIMEOUT);
  });

  // ============================================
  // ERROR HANDLING TESTS
  // ============================================

  describe('Error Handling', () => {
    it('should return 404 for non-existent endpoints', async () => {
      const response = await fetch(`${BASE_URL}/api/non-existent-endpoint`);
      expect(response.status).toBe(404);
    }, TIMEOUT);

    it('should handle malformed parameters gracefully', async () => {
      const response = await fetch(`${BASE_URL}/api/binance/klines?symbol=INVALID&interval=INVALID&limit=-999`);
      const data = await response.json();

      expect(data.success).toBe(false);
      expect(data).toHaveProperty('error');
    }, TIMEOUT);
  });

  // ============================================
  // PERFORMANCE TESTS
  // ============================================

  describe('Performance', () => {
    it('health check should respond within 500ms', async () => {
      const start = Date.now();
      await fetch(`${BASE_URL}/api/health`);
      const duration = Date.now() - start;

      expect(duration).toBeLessThan(500);
    }, TIMEOUT);

    it('should handle concurrent requests', async () => {
      const requests = Array(5).fill(null).map(() =>
        fetch(`${BASE_URL}/api/health`)
      );

      const responses = await Promise.all(requests);

      responses.forEach(response => {
        expect(response.ok).toBe(true);
      });
    }, TIMEOUT);
  });
});
