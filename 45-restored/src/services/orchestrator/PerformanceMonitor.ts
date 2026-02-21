export interface PerformanceMetrics {
  operation: string;
  duration: number;
  timestamp: number;
  success: boolean;
  metadata?: Record<string, any>;
}

export class PerformanceMonitor {
  private metrics: PerformanceMetrics[] = [];
  private readonly maxMetrics: number = 10000;

  recordMetric(metric: PerformanceMetrics): void {
    this.metrics.push(metric);

    if (this.metrics.length > this.maxMetrics) {
      this.metrics.shift();
    }
  }

  async measureAsync<T>(
    operation: string,
    fn: () => Promise<T>,
    metadata?: Record<string, any>
  ): Promise<T> {
    const startTime = Date.now();
    let success = false;

    try {
      const result = await fn();
      success = true;
      return result;
    } catch (error) {
      success = false;
      throw error;
    } finally {
      const duration = Date.now() - startTime;
      
      this.recordMetric({
        operation,
        duration,
        timestamp: startTime,
        success,
        metadata
      });
    }
  }

  measureSync<T>(
    operation: string,
    fn: () => T,
    metadata?: Record<string, any>
  ): T {
    const startTime = Date.now();
    let success = false;

    try {
      const result = fn();
      success = true;
      return result;
    } catch (error) {
      success = false;
      throw error;
    } finally {
      const duration = Date.now() - startTime;
      
      this.recordMetric({
        operation,
        duration,
        timestamp: startTime,
        success,
        metadata
      });
    }
  }

  getMetrics(operation?: string): PerformanceMetrics[] {
    if (operation) {
      return this.metrics.filter(m => m.operation === operation);
    }
    return [...this.metrics];
  }

  getStats(operation?: string): {
    count: number;
    avgDuration: number;
    minDuration: number;
    maxDuration: number;
    successRate: number;
    p50: number;
    p95: number;
    p99: number;
  } {
    const relevantMetrics = operation
      ? this.metrics.filter(m => m.operation === operation)
      : this.metrics;

    if (relevantMetrics.length === 0) {
      return {
        count: 0,
        avgDuration: 0,
        minDuration: 0,
        maxDuration: 0,
        successRate: 0,
        p50: 0,
        p95: 0,
        p99: 0
      };
    }

    const durations = relevantMetrics.map(m => m.duration).sort((a, b) => a - b);
    const successCount = relevantMetrics.filter(m => m.success).length;

    const p50Index = Math.floor(durations.length * 0.5);
    const p95Index = Math.floor(durations.length * 0.95);
    const p99Index = Math.floor(durations.length * 0.99);

    return {
      count: relevantMetrics.length,
      avgDuration: durations.reduce((a, b) => a + b, 0) / durations.length,
      minDuration: durations[0],
      maxDuration: durations[durations.length - 1],
      successRate: (successCount / relevantMetrics.length) * 100,
      p50: durations[p50Index],
      p95: durations[p95Index],
      p99: durations[p99Index]
    };
  }

  getRecentMetrics(count: number = 100): PerformanceMetrics[] {
    return this.metrics.slice(-count);
  }

  getOperations(): string[] {
    const operations = new Set(this.metrics.map(m => m.operation));
    return Array.from(operations);
  }

  clearMetrics(): void {
    this.metrics = [];
  }

  getReport(): {
    totalOperations: number;
    uniqueOperations: number;
    overallSuccessRate: number;
    operationStats: Array<{
      operation: string;
      count: number;
      avgDuration: number;
      successRate: number;
    }>;
  } {
    const operations = this.getOperations();
    const operationStats = operations.map(op => {
      const stats = this.getStats(op);
      return {
        operation: op,
        count: stats.count,
        avgDuration: stats.avgDuration,
        successRate: stats.successRate
      };
    });

    const totalSuccessful = this.metrics.filter(m => m.success).length;
    const overallSuccessRate = this.metrics.length > 0
      ? (totalSuccessful / this.metrics.length) * 100
      : 0;

    return {
      totalOperations: this.metrics.length,
      uniqueOperations: operations.length,
      overallSuccessRate,
      operationStats: operationStats.sort((a, b) => b.count - a.count)
    };
  }
}

let monitorInstance: PerformanceMonitor | null = null;

export function getPerformanceMonitor(): PerformanceMonitor {
  if (!monitorInstance) {
    monitorInstance = new PerformanceMonitor();
  }
  return monitorInstance;
}
