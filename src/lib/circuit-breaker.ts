/**
 * CIRCUIT BREAKER PATTERN
 * Prevents cascading failures and provides graceful degradation
 * Beyaz Şapka: Educational purpose only - resilience pattern
 */

export enum CircuitState {
  CLOSED = 'CLOSED',     // Normal operation
  OPEN = 'OPEN',         // Failing, reject requests
  HALF_OPEN = 'HALF_OPEN' // Testing if service recovered
}

export interface CircuitBreakerOptions {
  failureThreshold: number;    // Number of failures before opening circuit
  successThreshold: number;    // Number of successes to close circuit from half-open
  timeout: number;             // Time in ms to wait before trying again (half-open)
  resetTimeout?: number;       // Time in ms to auto-reset failure count
}

export interface CircuitBreakerStats {
  state: CircuitState;
  failures: number;
  successes: number;
  totalRequests: number;
  totalFailures: number;
  totalSuccesses: number;
  lastFailureTime: number | null;
  lastSuccessTime: number | null;
  nextAttemptTime: number | null;
}

export class CircuitBreaker {
  private state: CircuitState = CircuitState.CLOSED;
  private failures: number = 0;
  private successes: number = 0;
  private totalRequests: number = 0;
  private totalFailures: number = 0;
  private totalSuccesses: number = 0;
  private lastFailureTime: number | null = null;
  private lastSuccessTime: number | null = null;
  private nextAttemptTime: number | null = null;
  private resetTimer: NodeJS.Timeout | null = null;

  constructor(
    private name: string,
    private options: CircuitBreakerOptions = {
      failureThreshold: 5,
      successThreshold: 2,
      timeout: 60000, // 1 minute
      resetTimeout: 300000, // 5 minutes
    }
  ) {
    console.log(`[CircuitBreaker: ${name}] Initialized with`, options);
  }

  /**
   * Execute a function with circuit breaker protection
   */
  async execute<T>(fn: () => Promise<T>, fallback?: () => T): Promise<T> {
    this.totalRequests++;

    // Check if circuit is open
    if (this.state === CircuitState.OPEN) {
      const now = Date.now();

      if (this.nextAttemptTime && now < this.nextAttemptTime) {
        console.log(`[CircuitBreaker: ${this.name}] 🔴 Circuit OPEN - rejecting request`);

        if (fallback) {
          return fallback();
        }

        throw new Error(
          `Circuit breaker is OPEN for ${this.name}. ` +
          `Next attempt in ${Math.ceil((this.nextAttemptTime - now) / 1000)}s`
        );
      }

      // Time to try again - move to half-open
      console.log(`[CircuitBreaker: ${this.name}] 🟡 Moving to HALF_OPEN state`);
      this.state = CircuitState.HALF_OPEN;
      this.successes = 0;
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();

      if (fallback) {
        console.log(`[CircuitBreaker: ${this.name}] Using fallback function`);
        return fallback();
      }

      throw error;
    }
  }

  /**
   * Handle successful execution
   */
  private onSuccess(): void {
    this.totalSuccesses++;
    this.lastSuccessTime = Date.now();
    this.failures = 0; // Reset failure count on success

    if (this.state === CircuitState.HALF_OPEN) {
      this.successes++;
      console.log(
        `[CircuitBreaker: ${this.name}] ✅ Success ${this.successes}/${this.options.successThreshold} ` +
        `in HALF_OPEN state`
      );

      if (this.successes >= this.options.successThreshold) {
        this.close();
      }
    } else if (this.state === CircuitState.CLOSED) {
      // Schedule reset of failure count after resetTimeout
      if (this.options.resetTimeout && !this.resetTimer) {
        this.resetTimer = setTimeout(() => {
          if (this.failures > 0) {
            console.log(`[CircuitBreaker: ${this.name}] 🔄 Auto-resetting failure count`);
            this.failures = 0;
          }
          this.resetTimer = null;
        }, this.options.resetTimeout);
      }
    }
  }

  /**
   * Handle failed execution
   */
  private onFailure(): void {
    this.totalFailures++;
    this.failures++;
    this.lastFailureTime = Date.now();

    console.log(
      `[CircuitBreaker: ${this.name}] ❌ Failure ${this.failures}/${this.options.failureThreshold} ` +
      `(state: ${this.state})`
    );

    if (this.state === CircuitState.HALF_OPEN) {
      // Immediately open on failure in half-open state
      this.open();
    } else if (this.failures >= this.options.failureThreshold) {
      this.open();
    }

    // Clear reset timer on failure
    if (this.resetTimer) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }
  }

  /**
   * Open the circuit (reject requests)
   */
  private open(): void {
    this.state = CircuitState.OPEN;
    this.nextAttemptTime = Date.now() + this.options.timeout;

    console.log(
      `[CircuitBreaker: ${this.name}] 🔴 Circuit OPENED - ` +
      `will retry in ${this.options.timeout / 1000}s`
    );
  }

  /**
   * Close the circuit (normal operation)
   */
  private close(): void {
    this.state = CircuitState.CLOSED;
    this.failures = 0;
    this.successes = 0;
    this.nextAttemptTime = null;

    console.log(`[CircuitBreaker: ${this.name}] 🟢 Circuit CLOSED - resuming normal operation`);
  }

  /**
   * Get current circuit breaker statistics
   */
  public getStats(): CircuitBreakerStats {
    return {
      state: this.state,
      failures: this.failures,
      successes: this.successes,
      totalRequests: this.totalRequests,
      totalFailures: this.totalFailures,
      totalSuccesses: this.totalSuccesses,
      lastFailureTime: this.lastFailureTime,
      lastSuccessTime: this.lastSuccessTime,
      nextAttemptTime: this.nextAttemptTime,
    };
  }

  /**
   * Get success rate (0-100)
   */
  public getSuccessRate(): number {
    if (this.totalRequests === 0) return 100;
    return (this.totalSuccesses / this.totalRequests) * 100;
  }

  /**
   * Manually reset the circuit breaker
   */
  public reset(): void {
    console.log(`[CircuitBreaker: ${this.name}] 🔄 Manual reset`);
    this.state = CircuitState.CLOSED;
    this.failures = 0;
    this.successes = 0;
    this.nextAttemptTime = null;

    if (this.resetTimer) {
      clearTimeout(this.resetTimer);
      this.resetTimer = null;
    }
  }

  /**
   * Check if circuit is currently operational
   */
  public isOperational(): boolean {
    return this.state === CircuitState.CLOSED ||
           (this.state === CircuitState.HALF_OPEN &&
            !!this.nextAttemptTime && Date.now() >= this.nextAttemptTime);
  }
}

/**
 * Circuit Breaker Manager - singleton for managing multiple circuit breakers
 */
class CircuitBreakerManager {
  private breakers: Map<string, CircuitBreaker> = new Map();

  /**
   * Get or create a circuit breaker
   */
  public getBreaker(name: string, options?: CircuitBreakerOptions): CircuitBreaker {
    if (!this.breakers.has(name)) {
      this.breakers.set(name, new CircuitBreaker(name, options));
    }
    return this.breakers.get(name)!;
  }

  /**
   * Get all circuit breaker statistics
   */
  public getAllStats(): Record<string, CircuitBreakerStats> {
    const stats: Record<string, CircuitBreakerStats> = {};
    this.breakers.forEach((breaker, name) => {
      stats[name] = breaker.getStats();
    });
    return stats;
  }

  /**
   * Reset all circuit breakers
   */
  public resetAll(): void {
    console.log('[CircuitBreakerManager] Resetting all circuit breakers');
    this.breakers.forEach(breaker => breaker.reset());
  }
}

// Singleton instance
export const circuitBreakerManager = new CircuitBreakerManager();

/**
 * Helper function to create circuit breaker for API calls
 */
export function withCircuitBreaker<T>(
  name: string,
  fn: () => Promise<T>,
  options?: CircuitBreakerOptions,
  fallback?: () => T
): Promise<T> {
  const breaker = circuitBreakerManager.getBreaker(name, options);
  return breaker.execute(fn, fallback);
}
