/**
 * Performance Monitoring
 *
 * White-hat compliance: Tracks legitimate performance metrics
 * for improving user experience
 */

interface PerformanceMetric {
  name: string;
  value: number;
  unit: 'ms' | 'bytes' | 'count';
  timestamp: number;
  metadata?: Record<string, any>;
}

interface PerformanceReport {
  metrics: PerformanceMetric[];
  summary: {
    avgResponseTime: number;
    p95ResponseTime: number;
    p99ResponseTime: number;
    totalRequests: number;
    errorRate: number;
  };
  period: {
    start: number;
    end: number;
  };
}

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private readonly maxMetrics = 1000;

  /**
   * Record a performance metric
   */
  record(name: string, value: number, unit: 'ms' | 'bytes' | 'count' = 'ms', metadata?: Record<string, any>): void {
    const metric: PerformanceMetric = {
      name,
      value,
      unit,
      timestamp: Date.now(),
      metadata,
    };

    this.metrics.push(metric);

    // Keep only last N metrics
    if (this.metrics.length > this.maxMetrics) {
      this.metrics = this.metrics.slice(-this.maxMetrics);
    }
  }

  /**
   * Measure execution time of a function
   */
  async measure<T>(name: string, fn: () => Promise<T> | T): Promise<T> {
    const start = performance.now();
    try {
      const result = await fn();
      const duration = performance.now() - start;
      this.record(name, duration, 'ms');
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      this.record(name, duration, 'ms', { error: true });
      throw error;
    }
  }

  /**
   * Get metrics by name
   */
  getMetrics(name: string): PerformanceMetric[] {
    return this.metrics.filter(m => m.name === name);
  }

  /**
   * Get all metrics
   */
  getAllMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  /**
   * Get performance report
   */
  getReport(periodMs: number = 3600000): PerformanceReport {
    const now = Date.now();
    const start = now - periodMs;

    const periodMetrics = this.metrics.filter(m => m.timestamp >= start);

    // Calculate response time metrics
    const responseTimes = periodMetrics
      .filter(m => m.unit === 'ms')
      .map(m => m.value)
      .sort((a, b) => a - b);

    const avgResponseTime = responseTimes.length > 0
      ? responseTimes.reduce((sum, val) => sum + val, 0) / responseTimes.length
      : 0;

    const p95Index = Math.floor(responseTimes.length * 0.95);
    const p99Index = Math.floor(responseTimes.length * 0.99);

    const p95ResponseTime = responseTimes[p95Index] || 0;
    const p99ResponseTime = responseTimes[p99Index] || 0;

    // Calculate error rate
    const totalRequests = periodMetrics.filter(m => m.name.includes('request')).length;
    const errorRequests = periodMetrics.filter(m => m.metadata?.error === true).length;
    const errorRate = totalRequests > 0 ? errorRequests / totalRequests : 0;

    return {
      metrics: periodMetrics,
      summary: {
        avgResponseTime: Math.round(avgResponseTime * 100) / 100,
        p95ResponseTime: Math.round(p95ResponseTime * 100) / 100,
        p99ResponseTime: Math.round(p99ResponseTime * 100) / 100,
        totalRequests,
        errorRate: Math.round(errorRate * 10000) / 100, // Percentage
      },
      period: {
        start,
        end: now,
      },
    };
  }

  /**
   * Get Web Vitals
   */
  getWebVitals(): {
    cls: number | null;
    fid: number | null;
    lcp: number | null;
    fcp: number | null;
    ttfb: number | null;
  } {
    if (typeof window === 'undefined') {
      return { cls: null, fid: null, lcp: null, fcp: null, ttfb: null };
    }

    // Try to get performance entries
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType('paint');

    const ttfb = navigation ? navigation.responseStart - navigation.requestStart : null;
    const fcp = paint.find(p => p.name === 'first-contentful-paint')?.startTime || null;

    return {
      cls: null, // Cumulative Layout Shift - requires browser API
      fid: null, // First Input Delay - requires browser API
      lcp: null, // Largest Contentful Paint - requires browser API
      fcp,
      ttfb,
    };
  }

  /**
   * Clear old metrics
   */
  clearOldMetrics(olderThanMs: number = 3600000): number {
    const cutoff = Date.now() - olderThanMs;
    const initialCount = this.metrics.length;
    this.metrics = this.metrics.filter(m => m.timestamp > cutoff);
    return initialCount - this.metrics.length;
  }

  /**
   * Get metric statistics
   */
  getStats(name: string): {
    count: number;
    min: number;
    max: number;
    avg: number;
    median: number;
  } | null {
    const metrics = this.getMetrics(name);
    if (metrics.length === 0) return null;

    const values = metrics.map(m => m.value).sort((a, b) => a - b);

    return {
      count: values.length,
      min: values[0],
      max: values[values.length - 1],
      avg: values.reduce((sum, val) => sum + val, 0) / values.length,
      median: values[Math.floor(values.length / 2)],
    };
  }

  /**
   * Export metrics for external monitoring
   */
  exportMetrics(format: 'json' | 'prometheus' = 'json'): string {
    if (format === 'prometheus') {
      // Prometheus format
      const grouped = this.metrics.reduce((acc, m) => {
        if (!acc[m.name]) acc[m.name] = [];
        acc[m.name].push(m);
        return acc;
      }, {} as Record<string, PerformanceMetric[]>);

      let output = '';
      for (const [name, metrics] of Object.entries(grouped)) {
        const avg = metrics.reduce((sum, m) => sum + m.value, 0) / metrics.length;
        output += `# HELP ${name} Performance metric\n`;
        output += `# TYPE ${name} gauge\n`;
        output += `${name} ${avg}\n\n`;
      }
      return output;
    }

    // JSON format (default)
    return JSON.stringify(this.metrics, null, 2);
  }
}

// Singleton instance
const performanceMonitor = new PerformanceMonitor();

// Auto-clear old metrics every hour
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    performanceMonitor.clearOldMetrics();
  }, 3600000);
}

export default performanceMonitor;
export { PerformanceMonitor };
export type { PerformanceMetric, PerformanceReport };
