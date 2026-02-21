/**
 * CIRCUIT BREAKER PATTERN
 * Protects services from cascading failures
 *
 * States:
 * - CLOSED: Normal operation, requests pass through
 * - OPEN: Too many failures, requests fail fast
 * - HALF_OPEN: Testing if service recovered
 *
 * White-hat compliance: All state changes are logged
 */

export enum CircuitState {
  CLOSED = 'CLOSED',
  OPEN = 'OPEN',
  HALF_OPEN = 'HALF_OPEN',
}

export interface CircuitBreakerConfig {
  failureThreshold: number; // Number of failures before opening
  successThreshold: number; // Number of successes to close from half-open
  timeout: number; // Time in ms to wait before trying half-open
  monitoringPeriod: number; // Time window for failure counting
}

export interface CircuitBreakerStats {
  state: CircuitState;
  failures: number;
  successes: number;
  lastFailureTime?: number;
  lastSuccessTime?: number;
  nextAttemptTime?: number;
  totalCalls: number;
  totalFailures: number;
  totalSuccesses: number;
}

/**
 * Circuit Breaker Implementation
 */
export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failureCount: number = 0;
  private successCount: number = 0;
  private lastFailureTime?: number;
  private lastSuccessTime?: number;
  private nextAttemptTime?: number;
  private totalCalls: number = 0;
  private totalFailures: number = 0;
  private totalSuccesses: number = 0;

  constructor(
    private readonly name: string,
    private readonly config: CircuitBreakerConfig
  ) {
    console.log(`[CircuitBreaker] ${name} initialized:`, config);
  }

  /**
   * Execute a function with circuit breaker protection
   */
  async execute<T>(fn: () => Promise<T>, fallback?: () => Promise<T>): Promise<T> {
    this.totalCalls++;

    // Check if circuit is open
    if (this.state === CircuitState.OPEN) {
      if (this.shouldAttemptReset()) {
        console.log(`[CircuitBreaker] ${this.name} transitioning to HALF_OPEN`);
        this.state = CircuitState.HALF_OPEN;
        this.successCount = 0;
      } else {
        console.warn(
          `[CircuitBreaker] ${this.name} is OPEN, failing fast (next attempt: ${new Date(
            this.nextAttemptTime!
          ).toISOString()})`
        );

        if (fallback) {
          console.log(`[CircuitBreaker] ${this.name} using fallback`);
          return await fallback();
        }

        throw new Error(`Circuit breaker ${this.name} is OPEN`);
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error: any) {
      this.onFailure();

      // Use fallback if available
      if (fallback) {
        console.log(`[CircuitBreaker] ${this.name} error, using fallback:`, error.message);
        return await fallback();
      }

      throw error;
    }
  }

  /**
   * Record a successful call
   */
  private onSuccess(): void {
    this.totalSuccesses++;
    this.lastSuccessTime = Date.now();

    if (this.state === CircuitState.HALF_OPEN) {
      this.successCount++;

      if (this.successCount >= this.config.successThreshold) {
        console.log(`[CircuitBreaker] ${this.name} transitioning to CLOSED (recovered)`);
        this.state = CircuitState.CLOSED;
        this.failureCount = 0;
        this.successCount = 0;
      }
    } else if (this.state === CircuitState.CLOSED) {
      // Reset failure count on success
      this.failureCount = 0;
    }
  }

  /**
   * Record a failed call
   */
  private onFailure(): void {
    this.totalFailures++;
    this.lastFailureTime = Date.now();
    this.failureCount++;

    if (this.state === CircuitState.HALF_OPEN) {
      // Any failure in half-open state opens the circuit again
      console.warn(`[CircuitBreaker] ${this.name} transitioning to OPEN (half-open test failed)`);
      this.trip();
    } else if (this.state === CircuitState.CLOSED) {
      if (this.failureCount >= this.config.failureThreshold) {
        console.warn(
          `[CircuitBreaker] ${this.name} transitioning to OPEN (threshold ${this.config.failureThreshold} reached)`
        );
        this.trip();
      }
    }
  }

  /**
   * Open the circuit (fail fast mode)
   */
  private trip(): void {
    this.state = CircuitState.OPEN;
    this.nextAttemptTime = Date.now() + this.config.timeout;
    this.failureCount = 0;
    this.successCount = 0;
  }

  /**
   * Check if we should attempt to reset the circuit
   */
  private shouldAttemptReset(): boolean {
    return this.nextAttemptTime !== undefined && Date.now() >= this.nextAttemptTime;
  }

  /**
   * Get current circuit breaker stats
   */
  getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failures: this.failureCount,
      successes: this.successCount,
      lastFailureTime: this.lastFailureTime,
      lastSuccessTime: this.lastSuccessTime,
      nextAttemptTime: this.nextAttemptTime,
      totalCalls: this.totalCalls,
      totalFailures: this.totalFailures,
      totalSuccesses: this.totalSuccesses,
    };
  }

  /**
   * Get current state
   */
  getState(): CircuitState {
    return this.state;
  }

  /**
   * Manually reset the circuit breaker
   */
  reset(): void {
    console.log(`[CircuitBreaker] ${this.name} manually reset`);
    this.state = CircuitState.CLOSED;
    this.failureCount = 0;
    this.successCount = 0;
    this.nextAttemptTime = undefined;
  }

  /**
   * Check if circuit is healthy (closed or half-open)
   */
  isHealthy(): boolean {
    return this.state === CircuitState.CLOSED || this.state === CircuitState.HALF_OPEN;
  }
}

/**
 * Circuit Breaker Manager
 * Manages multiple circuit breakers
 */
export class CircuitBreakerManager {
  private breakers: Map<string, CircuitBreaker> = new Map();

  /**
   * Get or create a circuit breaker
   */
  getBreaker(name: string, config?: Partial<CircuitBreakerConfig>): CircuitBreaker {
    if (!this.breakers.has(name)) {
      const defaultConfig: CircuitBreakerConfig = {
        failureThreshold: 5,
        successThreshold: 2,
        timeout: 60000, // 1 minute
        monitoringPeriod: 60000, // 1 minute
        ...config,
      };

      this.breakers.set(name, new CircuitBreaker(name, defaultConfig));
    }

    return this.breakers.get(name)!;
  }

  /**
   * Get all circuit breaker stats
   */
  getAllStats(): Record<string, CircuitBreakerStats> {
    const stats: Record<string, CircuitBreakerStats> = {};

    for (const [name, breaker] of this.breakers.entries()) {
      stats[name] = breaker.getStats();
    }

    return stats;
  }

  /**
   * Health check for all breakers
   */
  getHealth(): {
    healthy: boolean;
    breakers: Record<string, { state: string; healthy: boolean }>;
  } {
    const breakers: Record<string, { state: string; healthy: boolean }> = {};
    let allHealthy = true;

    for (const [name, breaker] of this.breakers.entries()) {
      const healthy = breaker.isHealthy();
      breakers[name] = {
        state: breaker.getState(),
        healthy,
      };

      if (!healthy) {
        allHealthy = false;
      }
    }

    return { healthy: allHealthy, breakers };
  }

  /**
   * Reset all circuit breakers
   */
  resetAll(): void {
    console.log('[CircuitBreakerManager] Resetting all circuit breakers');
    for (const breaker of this.breakers.values()) {
      breaker.reset();
    }
  }

  /**
   * Reset a specific circuit breaker
   */
  reset(name: string): void {
    const breaker = this.breakers.get(name);
    if (breaker) {
      breaker.reset();
    }
  }
}

// Singleton instance
const circuitBreakerManager = new CircuitBreakerManager();
export default circuitBreakerManager;
