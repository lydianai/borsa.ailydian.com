/**
 * QUANTUM PRO SETTINGS UTILITY
 * LocalStorage-based settings management
 * WHITE HAT: Educational configuration only
 */

export interface QuantumProSettings {
  signals: {
    minConfidence: number;
    refreshInterval: number;
    maxSignals: number;
  };
  backtest: {
    defaultPeriod: string;
    showAllStrategies: boolean;
  };
  risk: {
    maxPositionSize: number;
    stopLossPercent: number;
    takeProfitPercent: number;
    dailyLossLimit: number;
  };
  bots: {
    autoStart: boolean;
    maxConcurrentBots: number;
  };
  monitoring: {
    refreshInterval: number;
    showLivePositions: boolean;
  };
}

const STORAGE_KEY = 'ailydian_quantum_pro_settings';

export const DEFAULT_SETTINGS: QuantumProSettings = {
  signals: {
    minConfidence: 0.60,
    refreshInterval: 30,
    maxSignals: 50,
  },
  backtest: {
    defaultPeriod: '30days',
    showAllStrategies: true,
  },
  risk: {
    maxPositionSize: 2,
    stopLossPercent: 1.5,
    takeProfitPercent: 3.0,
    dailyLossLimit: 5.0,
  },
  bots: {
    autoStart: false,
    maxConcurrentBots: 5,
  },
  monitoring: {
    refreshInterval: 5,
    showLivePositions: true,
  },
};

export function getQuantumSettings(): QuantumProSettings {
  if (typeof window === 'undefined') return DEFAULT_SETTINGS;

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
    }
  } catch (error) {
    console.error('Error loading Quantum Pro settings:', error);
  }

  return DEFAULT_SETTINGS;
}

export function saveQuantumSettings(settings: QuantumProSettings): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    window.dispatchEvent(new Event('quantumSettingsChanged'));
  } catch (error) {
    console.error('Error saving Quantum Pro settings:', error);
  }
}

export function resetQuantumSettings(): void {
  if (typeof window === 'undefined') return;

  try {
    localStorage.removeItem(STORAGE_KEY);
    window.dispatchEvent(new Event('quantumSettingsChanged'));
  } catch (error) {
    console.error('Error resetting Quantum Pro settings:', error);
  }
}
