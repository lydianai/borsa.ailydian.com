/**
 * Backend API Configuration
 *
 * White-hat compliance: Centralized configuration for backend services
 */

interface ServiceEndpoint {
  url: string;
  enabled: boolean;
  timeout: number;
}

interface BackendConfig {
  aiModels: ServiceEndpoint;
  taLib: ServiceEndpoint;
  database: ServiceEndpoint;
  monitoring: ServiceEndpoint;
}

const DEFAULT_TIMEOUT = 30000; // 30 seconds

const config: BackendConfig = {
  aiModels: {
    url: process.env.AI_MODELS_URL || 'http://localhost:5001',
    enabled: !!process.env.AI_MODELS_URL,
    timeout: DEFAULT_TIMEOUT,
  },
  taLib: {
    url: process.env.TALIB_SERVICE_URL || 'http://localhost:5002',
    enabled: !!process.env.TALIB_SERVICE_URL,
    timeout: DEFAULT_TIMEOUT,
  },
  database: {
    url: process.env.DATABASE_SERVICE_URL || 'http://localhost:5003',
    enabled: !!process.env.DATABASE_SERVICE_URL,
    timeout: DEFAULT_TIMEOUT,
  },
  monitoring: {
    url: process.env.MONITORING_SERVICE_URL || 'http://localhost:5004',
    enabled: !!process.env.MONITORING_SERVICE_URL,
    timeout: DEFAULT_TIMEOUT,
  },
};

/**
 * Get AI Models service endpoint
 */
export function getAIModelsEndpoint(): ServiceEndpoint {
  return config.aiModels;
}

/**
 * Get TA-Lib service endpoint
 */
export function getTALibEndpoint(): ServiceEndpoint {
  return config.taLib;
}

/**
 * Get Database service endpoint
 */
export function getDatabaseEndpoint(): ServiceEndpoint {
  return config.database;
}

/**
 * Get Monitoring service endpoint
 */
export function getMonitoringEndpoint(): ServiceEndpoint {
  return config.monitoring;
}

/**
 * Check if service is available
 */
export async function checkServiceHealth(endpoint: ServiceEndpoint): Promise<boolean> {
  if (!endpoint.enabled) return false;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);

    const response = await fetch(`${endpoint.url}/health`, {
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    return response.ok;
  } catch (error) {
    return false;
  }
}

/**
 * Get all service statuses
 */
export async function getAllServiceStatuses(): Promise<Record<string, boolean>> {
  const [aiModels, taLib, database, monitoring] = await Promise.all([
    checkServiceHealth(config.aiModels),
    checkServiceHealth(config.taLib),
    checkServiceHealth(config.database),
    checkServiceHealth(config.monitoring),
  ]);

  return {
    aiModels,
    taLib,
    database,
    monitoring,
  };
}

export default config;
export type { ServiceEndpoint, BackendConfig };
