/**
 * 🔧 SYSTEM HEALTH MONITOR
 * Arka plan servisleri ve sistem sağlığını izler
 * Hata/uyarıları Telegram'a bildirir
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Monitoring only (no trading)
 * - Educational purposes
 * - Error reporting only
 * - No sensitive data exposure
 */

import { broadcastMessage } from './notifications';
import {
  formatSystemError,
  formatBackgroundServiceError,
  formatAPIError,
  formatAnalysisError,
  formatDataQualityWarning,
  formatSystemHealthy,
} from './premium-formatter';

// ============================================================================
// TYPES
// ============================================================================

export interface ServiceHealth {
  name: string;
  status: 'healthy' | 'degraded' | 'down';
  lastCheck: Date;
  lastSuccess?: Date;
  lastError?: string;
  consecutiveFailures: number;
}

export interface SystemHealthReport {
  timestamp: Date;
  overallStatus: 'healthy' | 'degraded' | 'critical';
  services: ServiceHealth[];
  totalServices: number;
  healthyCount: number;
  degradedCount: number;
  downCount: number;
}

// ============================================================================
// SERVICE REGISTRY
// ============================================================================

const services: Map<string, ServiceHealth> = new Map();

/**
 * Servisi kaydet
 */
export function registerService(name: string): void {
  if (!services.has(name)) {
    services.set(name, {
      name,
      status: 'healthy',
      lastCheck: new Date(),
      consecutiveFailures: 0,
    });
  }
}

/**
 * Servis başarılı işlem kaydı
 */
export function recordServiceSuccess(name: string): void {
  const service = services.get(name);
  if (service) {
    service.status = 'healthy';
    service.lastCheck = new Date();
    service.lastSuccess = new Date();
    service.consecutiveFailures = 0;
    service.lastError = undefined;
  } else {
    // Otomatik kaydet
    registerService(name);
    recordServiceSuccess(name);
  }
}

/**
 * Servis hatası kaydı + Telegram bildirimi
 */
export async function recordServiceError(
  name: string,
  error: string,
  options?: {
    notify?: boolean; // Telegram'a bildir mi? (default: true)
    threshold?: number; // Kaç hata sonrası bildir? (default: 3)
  }
): Promise<void> {
  const notify = options?.notify !== false; // Default: true
  const threshold = options?.threshold ?? 3; // Default: 3 hatadan sonra bildir

  let service = services.get(name);

  if (!service) {
    registerService(name);
    service = services.get(name)!;
  }

  service.lastCheck = new Date();
  service.lastError = error;
  service.consecutiveFailures++;

  // Durumu güncelle
  if (service.consecutiveFailures >= threshold * 2) {
    service.status = 'down'; // Kritik
  } else if (service.consecutiveFailures >= threshold) {
    service.status = 'degraded'; // Düşük performans
  }

  // Telegram bildirimi (threshold aşıldıysa)
  if (notify && service.consecutiveFailures === threshold) {
    try {
      const message = formatBackgroundServiceError({
        name: service.name,
        error: service.lastError || 'Bilinmeyen hata',
        lastSuccessTime: service.lastSuccess,
      });

      await broadcastMessage(message, { parse_mode: 'HTML' });
    } catch (notifyError) {
      console.error('[System Monitor] Telegram bildirimi başarısız:', notifyError);
    }
  }

  // Kritik seviye: Tekrar bildir
  if (notify && service.consecutiveFailures === threshold * 2) {
    try {
      const message = formatSystemError({
        type: 'error',
        service: service.name,
        message: '🔴 KRİTİK: Servis tamamen çöktü',
        details: `${service.consecutiveFailures} ardışık hata\n${service.lastError}`,
      });

      await broadcastMessage(message, { parse_mode: 'HTML' });
    } catch (notifyError) {
      console.error('[System Monitor] Critical notification başarısız:', notifyError);
    }
  }
}

// ============================================================================
// API ERROR TRACKING
// ============================================================================

const apiErrors: Map<string, { count: number; lastError: string; lastNotified?: Date }> = new Map();

/**
 * API hatası kaydet + Telegram bildirimi
 */
export async function recordAPIError(
  endpoint: string,
  error: string,
  statusCode?: number,
  options?: {
    notify?: boolean;
    threshold?: number; // Kaç hata sonrası bildir? (default: 5)
  }
): Promise<void> {
  const notify = options?.notify !== false;
  const threshold = options?.threshold ?? 5;

  let apiError = apiErrors.get(endpoint);

  if (!apiError) {
    apiError = { count: 0, lastError: '' };
    apiErrors.set(endpoint, apiError);
  }

  apiError.count++;
  apiError.lastError = error;

  // Threshold aşıldıysa ve son 5 dk içinde bildirilmediyse
  const now = Date.now();
  const lastNotified = apiError.lastNotified?.getTime() ?? 0;
  const shouldNotify =
    notify && apiError.count >= threshold && now - lastNotified > 5 * 60 * 1000; // 5 dk

  if (shouldNotify) {
    try {
      const message = formatAPIError({
        endpoint,
        error,
        statusCode,
      });

      await broadcastMessage(message, { parse_mode: 'HTML' });

      apiError.lastNotified = new Date();
    } catch (notifyError) {
      console.error('[System Monitor] API error notification başarısız:', notifyError);
    }
  }
}

/**
 * API hata sayacını sıfırla (başarılı işlem sonrası)
 */
export function clearAPIErrors(endpoint: string): void {
  apiErrors.delete(endpoint);
}

// ============================================================================
// ANALYSIS ERROR TRACKING
// ============================================================================

const analysisErrors: Map<
  string,
  { count: number; lastError: string; lastNotified?: Date }
> = new Map();

/**
 * Analiz hatası kaydet + Telegram bildirimi
 */
export async function recordAnalysisError(
  strategy: string,
  symbol: string,
  error: string,
  options?: {
    notify?: boolean;
    threshold?: number; // Kaç hata sonrası bildir? (default: 10)
  }
): Promise<void> {
  const notify = options?.notify !== false;
  const threshold = options?.threshold ?? 10;

  const key = `${strategy}:${symbol}`;
  let analysisError = analysisErrors.get(key);

  if (!analysisError) {
    analysisError = { count: 0, lastError: '' };
    analysisErrors.set(key, analysisError);
  }

  analysisError.count++;
  analysisError.lastError = error;

  // Threshold aşıldıysa ve son 10 dk içinde bildirilmediyse
  const now = Date.now();
  const lastNotified = analysisError.lastNotified?.getTime() ?? 0;
  const shouldNotify =
    notify && analysisError.count >= threshold && now - lastNotified > 10 * 60 * 1000; // 10 dk

  if (shouldNotify) {
    try {
      const message = formatAnalysisError({
        strategy,
        symbol,
        error,
      });

      await broadcastMessage(message, { parse_mode: 'HTML' });

      analysisError.lastNotified = new Date();
    } catch (notifyError) {
      console.error('[System Monitor] Analysis error notification başarısız:', notifyError);
    }
  }
}

/**
 * Analiz hata sayacını sıfırla
 */
export function clearAnalysisErrors(strategy: string, symbol: string): void {
  const key = `${strategy}:${symbol}`;
  analysisErrors.delete(key);
}

// ============================================================================
// DATA QUALITY MONITORING
// ============================================================================

/**
 * Veri kalite uyarısı + Telegram bildirimi
 */
export async function recordDataQualityIssue(
  source: string,
  issue: string,
  affectedSymbols?: string[],
  options?: {
    notify?: boolean;
  }
): Promise<void> {
  const notify = options?.notify !== false;

  if (notify) {
    try {
      const message = formatDataQualityWarning({
        source,
        issue,
        affectedSymbols,
      });

      await broadcastMessage(message, { parse_mode: 'HTML' });
    } catch (notifyError) {
      console.error('[System Monitor] Data quality warning başarısız:', notifyError);
    }
  }
}

// ============================================================================
// SYSTEM HEALTH REPORTING
// ============================================================================

/**
 * Sistem sağlık raporu oluştur
 */
export function getSystemHealthReport(): SystemHealthReport {
  const servicesList = Array.from(services.values());

  const healthyCount = servicesList.filter((s) => s.status === 'healthy').length;
  const degradedCount = servicesList.filter((s) => s.status === 'degraded').length;
  const downCount = servicesList.filter((s) => s.status === 'down').length;

  let overallStatus: 'healthy' | 'degraded' | 'critical' = 'healthy';

  if (downCount > 0) {
    overallStatus = 'critical';
  } else if (degradedCount > 0 || healthyCount < servicesList.length * 0.8) {
    overallStatus = 'degraded';
  }

  return {
    timestamp: new Date(),
    overallStatus,
    services: servicesList,
    totalServices: servicesList.length,
    healthyCount,
    degradedCount,
    downCount,
  };
}

/**
 * Sağlık raporu Telegram'a gönder (sadece sorun varsa)
 */
export async function sendHealthReportIfNeeded(): Promise<void> {
  const report = getSystemHealthReport();

  // Sadece sorun varsa gönder
  if (report.overallStatus !== 'healthy') {
    const problemServices = report.services.filter((s) => s.status !== 'healthy');

    const message = formatSystemError({
      type: report.overallStatus === 'critical' ? 'error' : 'warning',
      service: 'System Health',
      message: `${report.healthyCount}/${report.totalServices} servis sağlıklı`,
      details: `Sorunlu servisler:\n${problemServices.map((s) => `- ${s.name}: ${s.status} (${s.consecutiveFailures} hata)`).join('\n')}`,
    });

    await broadcastMessage(message, { parse_mode: 'HTML' });
  }
}

/**
 * Günlük sistem sağlık özeti (her şey OK dahil)
 */
export async function sendDailyHealthSummary(): Promise<void> {
  const report = getSystemHealthReport();

  if (report.overallStatus === 'healthy') {
    const healthyServices = report.services.map((s) => s.name);
    const message = formatSystemHealthy(healthyServices);

    await broadcastMessage(message, { parse_mode: 'HTML' });
  } else {
    await sendHealthReportIfNeeded();
  }
}

// ============================================================================
// PERIODIC HEALTH CHECK (Cron Job)
// ============================================================================

let healthCheckInterval: NodeJS.Timeout | null = null;

/**
 * Periyodik sistem sağlık kontrolü başlat
 */
export function startHealthMonitoring(intervalMinutes: number = 30): void {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
  }

  // İlk kontrol
  sendHealthReportIfNeeded();

  // Periyodik kontrol
  healthCheckInterval = setInterval(
    () => {
      sendHealthReportIfNeeded();
    },
    intervalMinutes * 60 * 1000
  );

  console.log(`[System Monitor] Health monitoring started (${intervalMinutes} min intervals)`);
}

/**
 * Sağlık kontrolünü durdur
 */
export function stopHealthMonitoring(): void {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
    console.log('[System Monitor] Health monitoring stopped');
  }
}

// ============================================================================
// AUTO-REGISTER COMMON SERVICES
// ============================================================================

// Yaygın servisleri otomatik kaydet
registerService('Strategy Aggregator');
registerService('AI Bots');
registerService('Onchain Monitor');
registerService('Traditional Markets');
registerService('Correlation Analysis');
registerService('Futures Matrix');
registerService('Market Correlation');
registerService('Binance API');
registerService('Alpha Vantage API');
registerService('CoinGecko API');
registerService('Telegram Bot');

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  // Service Health
  registerService,
  recordServiceSuccess,
  recordServiceError,

  // API Errors
  recordAPIError,
  clearAPIErrors,

  // Analysis Errors
  recordAnalysisError,
  clearAnalysisErrors,

  // Data Quality
  recordDataQualityIssue,

  // Health Reporting
  getSystemHealthReport,
  sendHealthReportIfNeeded,
  sendDailyHealthSummary,

  // Monitoring Control
  startHealthMonitoring,
  stopHealthMonitoring,
};
