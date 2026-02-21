/**
 * Nirvana TF Bot Client
 * TypeScript client for Python FastAPI microservice
 * Integrates TensorFlow Python models with Next.js frontend
 */

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Indicators {
  rsi?: number;
  macd?: number;
  macd_signal?: number;
  macd_histogram?: number;
  bb_upper?: number;
  bb_middle?: number;
  bb_lower?: number;
  volume_ratio?: number;
  ema_short?: number;
  ema_long?: number;
  [key: string]: number | undefined;
}

interface SignalRequest {
  symbol: string;
  candles: Candle[];
  indicators: Indicators;
  timeframe?: string;
}

interface NirvanaSignal {
  decision: 'BUY' | 'HOLD' | 'PASS';
  confidence: number;
  probability: number;
  model: string;
  reasoning: string[];
  base_probability: number;
  adjusted_confidence: number;
  indicator_adjustments: number;
  symbol?: string;
  timeframe?: string;
  timestamp?: number;
  currentPrice?: number;
}

interface BatchSignalRequest {
  requests: SignalRequest[];
}

interface BatchSignalResponse {
  success: boolean;
  signals: NirvanaSignal[];
  errors?: Array<{ symbol: string; error: string }>;
  totalProcessed: number;
  successCount: number;
  errorCount: number;
}

interface HealthResponse {
  status: string;
  model: string;
  version: string;
  tensorflow_version: string;
}

interface ModelInfo {
  architecture: string;
  inputShape: number[];
  outputShape: number[];
  activation: string;
  optimizer: string;
  loss: string;
  summary: string;
}

export class NirvanaTFClient {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = 'http://localhost:8000', timeout: number = 30000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  /**
   * Check if Nirvana TF service is healthy
   */
  async healthCheck(): Promise<HealthResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'GET',
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Nirvana TF service timeout (5s)');
      }
      throw new Error(`Nirvana TF service unavailable: ${error}`);
    }
  }

  /**
   * Generate trading signal for a single symbol
   */
  async generateSignal(request: SignalRequest): Promise<NirvanaSignal> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(`${this.baseUrl}/signal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      return data.signal;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Nirvana TF prediction timeout (${this.timeout}ms)`);
      }
      throw error;
    }
  }

  /**
   * Generate trading signals for multiple symbols (batch)
   */
  async generateBatchSignals(requests: SignalRequest[]): Promise<BatchSignalResponse> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout * 2);

      const response = await fetch(`${this.baseUrl}/signals/batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ requests }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error(`Nirvana TF batch prediction timeout (${this.timeout * 2}ms)`);
      }
      throw error;
    }
  }

  /**
   * Get model architecture information
   */
  async getModelInfo(): Promise<ModelInfo> {
    try {
      const response = await fetch(`${this.baseUrl}/model/info`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`Failed to get model info: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      throw new Error(`Failed to get model info: ${error}`);
    }
  }

  /**
   * Check if service is available
   */
  async isAvailable(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch {
      return false;
    }
  }
}

// Singleton instance
let nirvanaClient: NirvanaTFClient | null = null;

/**
 * Get Nirvana TF Client singleton
 */
export function getNirvanaClient(baseUrl?: string): NirvanaTFClient {
  if (!nirvanaClient) {
    const serviceUrl = baseUrl || process.env.NIRVANA_TF_URL || 'http://localhost:8000';
    nirvanaClient = new NirvanaTFClient(serviceUrl);
  }
  return nirvanaClient;
}

/**
 * Reset client (for testing)
 */
export function resetNirvanaClient(): void {
  nirvanaClient = null;
}