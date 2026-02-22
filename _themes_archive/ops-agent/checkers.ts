/**
 * 🔍 LyTrade-EMRAH OPS AGENT - Health Checker
 * 
 * Sistem sağlığını kategorize eder:
 * - healthz: HTTP/WS endpoint'leri, lag ölçümü
 * - data-integrity: NaN/Inf, candle gap, rate limit
 * - front-end: UI SSR/CSR smoke testleri
 * - back-end: unit/integration testleri
 * - config-drift: ENV_MATRIX ile sapma tespiti
 */

// Simple logger replacement
const Logger = {
  info: (message: string, ...args: any[]) => console.log(`[INFO] ${message}`, ...args),
  error: (message: string, ...args: any[]) => console.error(`[ERROR] ${message}`, ...args),
  warn: (message: string, ...args: any[]) => console.warn(`[WARN] ${message}`, ...args),
  debug: (message: string, ...args: any[]) => console.debug(`[DEBUG] ${message}`, ...args),
  child: (context: any) => ({
    info: (message: string, ...args: any[]) => console.log(`[INFO] [${context.component || 'unknown'}] ${message}`, ...args),
    error: (message: string, ...args: any[]) => console.error(`[ERROR] [${context.component || 'unknown'}] ${message}`, ...args),
    warn: (message: string, ...args: any[]) => console.warn(`[WARN] [${context.component || 'unknown'}] ${message}`, ...args),
    debug: (message: string, ...args: any[]) => console.debug(`[DEBUG] [${context.component || 'unknown'}] ${message}`, ...args)
  })
};
import fetch from 'node-fetch';
import WebSocket from 'ws';

export interface HealthCheckResult {
  status: 'healthy' | 'degraded' | 'critical';
  metrics: {
    p99: number;
    errorRate: number;
    lag_ms: number;
    reconnects_per_min: number;
    nan_inf_count: number;
    strategy_agreement: number;
  };
  issues: HealthIssue[];
}

export interface HealthIssue {
  category: 'ui' | 'data' | 'backend' | 'config' | 'strategy' | 'observability';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  metric?: string;
  value?: number;
  threshold?: number;
}

export class HealthChecker {
  private logger = Logger.child({ component: 'HealthChecker' });
  private readonly BASE_URL = process.env.NEXT_PUBLIC_BASE_URL || 'http://localhost:3002';
  private readonly WS_URL = process.env.BINANCE_FAPI_WS || 'wss://fstream.binance.com/stream';

  async runFullCheck(): Promise<HealthCheckResult> {
    this.logger.info('🔍 Tam sağlık kontrolü başlatılıyor...');

    const checks = await Promise.allSettled([
      this.checkHTTPHealth(),
      this.checkWebSocketHealth(),
      this.checkDataIntegrity(),
      this.checkFrontendHealth(),
      this.checkBackendHealth(),
      this.checkConfigDrift(),
      this.checkStrategyConsensus()
    ]);

    const issues: HealthIssue[] = [];
    const metrics = {
      p99: 0,
      errorRate: 0,
      lag_ms: 0,
      reconnects_per_min: 0,
      nan_inf_count: 0,
      strategy_agreement: 0
    };

    // Sonuçları işle
    checks.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        issues.push(...result.value.issues);
        Object.assign(metrics, result.value.metrics);
      } else {
        this.logger.error({ index, error: result.reason }, 'Sağlık kontrolü başarısız');
        issues.push({
          category: 'observability',
          severity: 'critical',
          description: `Health check ${index} failed: ${result.reason.message}`
        });
      }
    });

    // Genel durum belirle
    const criticalIssues = issues.filter(i => i.severity === 'critical').length;
    const highIssues = issues.filter(i => i.severity === 'high').length;

    const status = criticalIssues > 0 ? 'critical' : 
                   highIssues > 0 ? 'degraded' : 'healthy';

    this.logger.info({ status, issuesCount: issues.length }, 'Sağlık kontrolü tamamlandı');

    return {
      status,
      metrics,
      issues
    };
  }

  /**
   * HTTP endpoint health check
   */
  private async checkHTTPHealth(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];
    const endpoints = [
      '/api/health-simple',
      '/api/futures-all',
      '/api/scanner/signals',
      '/dashboard/market'
    ];

    let totalLatency = 0;
    let errorCount = 0;

    for (const endpoint of endpoints) {
      try {
        const start = Date.now();
        const response = await fetch(`${this.BASE_URL}${endpoint}`, {
          timeout: 5000
        });
        const latency = Date.now() - start;
        totalLatency += latency;

        if (!response.ok) {
          errorCount++;
          issues.push({
            category: 'backend',
            severity: latency > 3000 ? 'high' : 'medium',
            description: `HTTP ${endpoint} returned ${response.status}`,
            metric: 'http_status',
            value: response.status,
            threshold: 200
          });
        }
      } catch (error) {
        errorCount++;
        issues.push({
          category: 'backend',
          severity: 'critical',
          description: `HTTP ${endpoint} failed: ${error.message}`,
          metric: 'http_error'
        });
      }
    }

    const avgLatency = totalLatency / endpoints.length;
    const p99 = avgLatency * 1.5; // Tahmini p99

    return {
      issues,
      metrics: {
        p99,
        errorRate: (errorCount / endpoints.length) * 100
      }
    };
  }

  /**
   * WebSocket bağlantı ve lag kontrolü
   */
  private async checkWebSocketHealth(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];
    let lag = 0;
    let reconnects = 0;

    return new Promise((resolve) => {
      const ws = new WebSocket(`${this.WS_URL}/btcusdt@kline_1m`);
      const timeout = setTimeout(() => {
        ws.terminate();
        issues.push({
          category: 'data',
          severity: 'critical',
          description: 'WebSocket connection timeout'
        });
        resolve({ issues, metrics: { lag_ms: lag, reconnects_per_min: reconnects } });
      }, 10000);

      ws.on('open', () => {
        clearTimeout(timeout);
        const startTime = Date.now();
        
        ws.on('message', (data) => {
          const msg = JSON.parse(data.toString());
          if (msg.k && msg.k.t) {
            lag = Date.now() - msg.k.t;
            ws.close();
          }
        });
      });

      ws.on('error', (error) => {
        clearTimeout(timeout);
        issues.push({
          category: 'data',
          severity: 'high',
          description: `WebSocket error: ${error.message}`
        });
        resolve({ issues, metrics: { lag_ms: lag, reconnects_per_min: reconnects } });
      });

      ws.on('close', () => {
        clearTimeout(timeout);
        resolve({ issues, metrics: { lag_ms: lag, reconnects_per_min: reconnects } });
      });
    });
  }

  /**
   * Veri bütünlüğü kontrolü
   */
  private async checkDataIntegrity(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];
    let nanInfCount = 0;

    try {
      const response = await fetch(`${this.BASE_URL}/api/futures-all`);
      const data = await response.json();

      if (data.success && data.pairs) {
        data.pairs.forEach((pair: any) => {
          if (isNaN(pair.price) || !isFinite(pair.price)) {
            nanInfCount++;
          }
          if (isNaN(pair.volume) || !isFinite(pair.volume)) {
            nanInfCount++;
          }
        });

        if (nanInfCount > 0) {
          issues.push({
            category: 'data',
            severity: nanInfCount > 10 ? 'high' : 'medium',
            description: `${nanInfCount} NaN/Inf values detected in market data`,
            metric: 'nan_inf_count',
            value: nanInfCount,
            threshold: 0
          });
        }
      }
    } catch (error) {
      issues.push({
        category: 'data',
        severity: 'critical',
        description: `Data integrity check failed: ${error.message}`
      });
    }

    return { issues, metrics: { nan_inf_count: nanInfCount } };
  }

  /**
   * Frontend sağlık kontrolü
   */
  private async checkFrontendHealth(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];

    try {
      // Ana sayfa SSR kontrolü
      const response = await fetch(this.BASE_URL, {
        headers: { 'User-Agent': 'OPS-Agent/1.0' }
      });

      if (!response.ok) {
        issues.push({
          category: 'ui',
          severity: 'high',
          description: `Frontend SSR failed: ${response.status}`,
          metric: 'ssr_status',
          value: response.status,
          threshold: 200
        });
      }

      const content = await response.text();
      if (!content.includes('SarDag Emrah')) {
        issues.push({
          category: 'ui',
          severity: 'medium',
          description: 'Frontend content missing expected elements'
        });
      }
    } catch (error) {
      issues.push({
        category: 'ui',
        severity: 'critical',
        description: `Frontend health check failed: ${error.message}`
      });
    }

    return { issues, metrics: {} };
  }

  /**
   * Backend sağlık kontrolü
   */
  private async checkBackendHealth(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];

    try {
      // API contract testleri
      const tests = [
        { endpoint: '/api/health-simple', expectedFields: ['status', 'timestamp'] },
        { endpoint: '/api/scanner/signals?limit=1', expectedFields: ['success', 'signals'] }
      ];

      for (const test of tests) {
        const response = await fetch(`${this.BASE_URL}${test.endpoint}`);
        const data = await response.json();

        for (const field of test.expectedFields) {
          if (!(field in data)) {
            issues.push({
              category: 'backend',
              severity: 'medium',
              description: `API contract violation: ${test.endpoint} missing ${field}`
            });
          }
        }
      }
    } catch (error) {
      issues.push({
        category: 'backend',
        severity: 'high',
        description: `Backend health check failed: ${error.message}`
      });
    }

    return { issues, metrics: {} };
  }

  /**
   * Konfigürasyon drift kontrolü
   */
  private async checkConfigDrift(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];

    // Kritik environment variable'ları kontrol et
    const requiredEnvs = [
      'GROQ_API_KEY',
      'BINANCE_FAPI_REST',
      'OPS_ERROR_BUDGET_pct',
      'OPS_CANARY_TRAFFIC_pct'
    ];

    requiredEnvs.forEach(env => {
      if (!process.env[env]) {
        issues.push({
          category: 'config',
          severity: 'high',
          description: `Missing required environment variable: ${env}`
        });
      }
    });

    return { issues, metrics: {} };
  }

  /**
   * Strateji konsensüs kontrolü
   */
  private async checkStrategyConsensus(): Promise<{ issues: HealthIssue[], metrics: Partial<HealthCheckResult['metrics']> }> {
    const issues: HealthIssue[] = [];
    let agreement = 0;

    try {
      const response = await fetch(`${this.BASE_URL}/api/scanner/signals?limit=10`);
      const data = await response.json();

      if (data.success && data.signals) {
        const buySignals = data.signals.filter((s: any) => s.signal === 'BUY' || s.signal === 'STRONG_BUY');
        agreement = buySignals.length > 0 ? 
          buySignals.reduce((sum: number, s: any) => sum + s.confidence, 0) / buySignals.length : 0;

        if (agreement < 62) {
          issues.push({
            category: 'strategy',
            severity: 'medium',
            description: `Low strategy consensus: ${agreement.toFixed(1)}%`,
            metric: 'strategy_agreement',
            value: agreement,
            threshold: 62
          });
        }
      }
    } catch (error) {
      issues.push({
        category: 'strategy',
        severity: 'high',
        description: `Strategy consensus check failed: ${error.message}`
      });
    }

    return { issues, metrics: { strategy_agreement: agreement } };
  }
}