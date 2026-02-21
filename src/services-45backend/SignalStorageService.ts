/**
 * Trading Signals Storage Service
 * Stores consensus approved signals for settings page
 */

interface StoredSignal {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  confidence: number;
  entry: number;
  stopLoss: number;
  takeProfit: number;
  timestamp: number;
  status: 'ACTIVE' | 'EXECUTED' | 'CANCELLED';
}

class SignalStorageService {
  private readonly STORAGE_KEY = 'consensus_approved_signals';
  private readonly MAX_SIGNALS = 100;

  // Store a new signal
  storeSignal(signal: {
    symbol: string;
    direction: 'LONG' | 'SHORT';
    confidence: number;
    entry: number;
    stopLoss: number;
    takeProfit: number;
  }): void {
    const signals = this.getAllSignals();
    
    const newSignal: StoredSignal = {
      id: `${signal.symbol}_${Date.now()}`,
      ...signal,
      timestamp: Date.now(),
      status: 'ACTIVE'
    };

    signals.unshift(newSignal);

    // Keep only last MAX_SIGNALS
    if (signals.length > this.MAX_SIGNALS) {
      signals.splice(this.MAX_SIGNALS);
    }

    this.saveSignals(signals);
  }

  // Get all signals
  getAllSignals(): StoredSignal[] {
    if (typeof window === 'undefined') return [];
    
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error reading signals:', error);
      return [];
    }
  }

  // Get active signals only
  getActiveSignals(): StoredSignal[] {
    return this.getAllSignals().filter(s => s.status === 'ACTIVE');
  }

  // Update signal status
  updateSignalStatus(id: string, status: 'ACTIVE' | 'EXECUTED' | 'CANCELLED'): void {
    const signals = this.getAllSignals();
    const signal = signals.find(s => s.id === id);
    
    if (signal) {
      signal.status = status;
      this.saveSignals(signals);
    }
  }

  // Delete a signal
  deleteSignal(id: string): void {
    const signals = this.getAllSignals().filter(s => s.id !== id);
    this.saveSignals(signals);
  }

  // Clear all signals
  clearAllSignals(): void {
    this.saveSignals([]);
  }

  // Get signals by symbol
  getSignalsBySymbol(symbol: string): StoredSignal[] {
    return this.getAllSignals().filter(s => s.symbol === symbol);
  }

  // Get statistics
  getStatistics() {
    const signals = this.getAllSignals();
    const active = signals.filter(s => s.status === 'ACTIVE').length;
    const executed = signals.filter(s => s.status === 'EXECUTED').length;
    const cancelled = signals.filter(s => s.status === 'CANCELLED').length;

    return {
      total: signals.length,
      active,
      executed,
      cancelled,
      successRate: executed > 0 ? (executed / (executed + cancelled)) * 100 : 0
    };
  }

  private saveSignals(signals: StoredSignal[]): void {
    if (typeof window === 'undefined') return;
    
    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(signals));
    } catch (error) {
      console.error('Error saving signals:', error);
    }
  }
}

export const signalStorage = new SignalStorageService();
export type { StoredSignal };
