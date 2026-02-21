export class CircuitBreaker {
  private failureCount: number = 0;
  private successCount: number = 0;
  private lastFailureTime: number = 0;
  private state: 'CLOSED' | 'OPEN' | 'HALF_OPEN' = 'CLOSED';

  constructor(
    private readonly name: string,
    private readonly threshold: number = 5,
    private readonly timeout: number = 60000,
    private readonly successThreshold: number = 2
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'OPEN') {
      if (Date.now() - this.lastFailureTime >= this.timeout) {
        console.log(`[CircuitBreaker:${this.name}] Transitioning to HALF_OPEN`);
        this.state = 'HALF_OPEN';
        this.successCount = 0;
      } else {
        throw new Error(`Circuit breaker OPEN for ${this.name} - please try again later`);
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;

    if (this.state === 'HALF_OPEN') {
      this.successCount++;
      
      if (this.successCount >= this.successThreshold) {
        console.log(`[CircuitBreaker:${this.name}] Transitioning to CLOSED`);
        this.state = 'CLOSED';
        this.successCount = 0;
      }
    }
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    this.successCount = 0;

    if (this.failureCount >= this.threshold) {
      console.error(`[CircuitBreaker:${this.name}] Transitioning to OPEN (failures: ${this.failureCount})`);
      this.state = 'OPEN';
    }
  }

  getState(): { state: string; failureCount: number; successCount: number } {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount
    };
  }

  reset(): void {
    this.state = 'CLOSED';
    this.failureCount = 0;
    this.successCount = 0;
    this.lastFailureTime = 0;
    console.log(`[CircuitBreaker:${this.name}] Reset to CLOSED`);
  }

  isOpen(): boolean {
    return this.state === 'OPEN';
  }
}

export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    initialDelay?: number;
    maxDelay?: number;
    backoffMultiplier?: number;
    onRetry?: (attempt: number, error: any) => void;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelay = 1000,
    maxDelay = 30000,
    backoffMultiplier = 2,
    onRetry
  } = options;

  let lastError: any;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries) {
        throw error;
      }

      const delay = Math.min(
        initialDelay * Math.pow(backoffMultiplier, attempt),
        maxDelay
      );

      if (onRetry) {
        onRetry(attempt + 1, error);
      }

      console.warn(`[Retry] Attempt ${attempt + 1}/${maxRetries} failed. Retrying in ${delay}ms...`);
      
      await sleep(delay);
    }
  }

  throw lastError;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export class BotFailoverManager {
  private failedBots: Map<string, { failureTime: number; attempts: number }> = new Map();
  private readonly maxRecoveryAttempts = 3;
  private readonly recoveryInterval = 300000;

  constructor(
    private readonly onBotFailed: (botId: string) => Promise<void>,
    private readonly onBotRecovered: (botId: string) => Promise<void>
  ) {}

  async handleBotFailure(botId: string, error: any): Promise<void> {
    console.error(`[Failover] Bot ${botId} failed:`, error);

    const failureInfo = this.failedBots.get(botId) || { failureTime: Date.now(), attempts: 0 };
    failureInfo.attempts++;
    failureInfo.failureTime = Date.now();
    this.failedBots.set(botId, failureInfo);

    await this.onBotFailed(botId);

    if (failureInfo.attempts < this.maxRecoveryAttempts) {
      setTimeout(async () => {
        await this.attemptRecovery(botId);
      }, this.recoveryInterval);
    } else {
      console.error(`[Failover] Bot ${botId} exceeded max recovery attempts (${this.maxRecoveryAttempts})`);
    }
  }

  private async attemptRecovery(botId: string): Promise<void> {
    const failureInfo = this.failedBots.get(botId);
    if (!failureInfo) return;

    console.log(`[Failover] Attempting recovery for bot ${botId} (attempt ${failureInfo.attempts}/${this.maxRecoveryAttempts})`);

    try {
      await this.onBotRecovered(botId);
      
      this.failedBots.delete(botId);
      console.log(`[Failover] Bot ${botId} recovered successfully`);
    } catch (error) {
      console.error(`[Failover] Recovery failed for bot ${botId}:`, error);
      
      if (failureInfo.attempts < this.maxRecoveryAttempts) {
        setTimeout(async () => {
          await this.attemptRecovery(botId);
        }, this.recoveryInterval);
      }
    }
  }

  getFailedBots(): string[] {
    return Array.from(this.failedBots.keys());
  }

  clearFailure(botId: string): void {
    this.failedBots.delete(botId);
    console.log(`[Failover] Cleared failure record for bot ${botId}`);
  }

  getFailureInfo(botId: string): { attempts: number; failureTime: number } | undefined {
    return this.failedBots.get(botId);
  }
}

export class RateLimiter {
  private requests: number[] = [];

  constructor(
    private readonly maxRequests: number,
    private readonly windowMs: number
  ) {}

  async acquire(): Promise<void> {
    const now = Date.now();
    
    this.requests = this.requests.filter(time => now - time < this.windowMs);

    if (this.requests.length >= this.maxRequests) {
      const oldestRequest = this.requests[0];
      const waitTime = this.windowMs - (now - oldestRequest);
      
      console.warn(`[RateLimiter] Rate limit reached. Waiting ${waitTime}ms...`);
      await sleep(waitTime);
      
      return this.acquire();
    }

    this.requests.push(now);
  }

  getStats(): { current: number; max: number; windowMs: number } {
    const now = Date.now();
    this.requests = this.requests.filter(time => now - time < this.windowMs);

    return {
      current: this.requests.length,
      max: this.maxRequests,
      windowMs: this.windowMs
    };
  }

  reset(): void {
    this.requests = [];
  }
}

export class ErrorTracker {
  private errors: Map<string, { count: number; lastError: any; lastOccurrence: number }> = new Map();

  track(context: string, error: any): void {
    const existing = this.errors.get(context) || { count: 0, lastError: null, lastOccurrence: 0 };
    
    existing.count++;
    existing.lastError = error;
    existing.lastOccurrence = Date.now();
    
    this.errors.set(context, existing);

    console.error(`[ErrorTracker:${context}] Error #${existing.count}:`, error);
  }

  getErrors(): Map<string, { count: number; lastError: any; lastOccurrence: number }> {
    return new Map(this.errors);
  }

  getErrorCount(context: string): number {
    return this.errors.get(context)?.count || 0;
  }

  clearErrors(context?: string): void {
    if (context) {
      this.errors.delete(context);
    } else {
      this.errors.clear();
    }
  }

  getTopErrors(limit: number = 10): Array<{ context: string; count: number; lastOccurrence: number }> {
    return Array.from(this.errors.entries())
      .map(([context, info]) => ({
        context,
        count: info.count,
        lastOccurrence: info.lastOccurrence
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, limit);
  }
}
