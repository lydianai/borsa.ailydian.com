/**
 * BACKTESTING ENGINE
 * Test AI signals on historical data
 * Metrics: Sharpe Ratio, Win Rate, Max Drawdown, Total Return
 */

interface BacktestTrade {
  entryTime: number;
  exitTime: number;
  entryPrice: number;
  exitPrice: number;
  action: 'BUY' | 'SELL';
  profit: number;
  profitPercent: number;
}

interface BacktestResult {
  symbol: string;
  startDate: string;
  endDate: string;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalReturn: number;
  totalReturnPercent: number;
  sharpeRatio: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  avgProfit: number;
  avgLoss: number;
  profitFactor: number;
  trades: BacktestTrade[];
}

export class BacktestingEngine {
  private readonly INITIAL_CAPITAL = 10000;
  private readonly POSITION_SIZE = 0.1; // 10% of capital per trade
  private readonly COMMISSION = 0.001; // 0.1% commission

  /**
   * Run backtest for a symbol
   */
  async runBacktest(
    symbol: string,
    startDate: Date,
    endDate: Date
  ): Promise<BacktestResult> {
    try {
      // 1. Fetch historical data
      const historicalData = await this.fetchHistoricalData(symbol, startDate, endDate);

      // 2. Generate signals for each data point
      const signals = await this.generateHistoricalSignals(symbol, historicalData);

      // 3. Simulate trades
      const trades = this.simulateTrades(historicalData, signals);

      // 4. Calculate metrics
      const metrics = this.calculateMetrics(trades);

      return {
        symbol,
        startDate: startDate.toISOString(),
        endDate: endDate.toISOString(),
        ...metrics,
        trades
      };

    } catch (error) {
      console.error(`Backtest error for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Fetch historical OHLCV data
   */
  private async fetchHistoricalData(
    _symbol: string,
    startDate: Date,
    endDate: Date
  ): Promise<any[]> {
    // In production, fetch from Binance/CoinGecko
    // For now, generate mock data
    const data: any[] = [];
    const days = Math.floor((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24));

    let price = 100; // Starting price

    for (let i = 0; i < days; i++) {
      const change = (Math.random() - 0.48) * 5; // Slight upward bias
      price += change;

      data.push({
        timestamp: startDate.getTime() + i * 24 * 60 * 60 * 1000,
        open: price,
        high: price * 1.02,
        low: price * 0.98,
        close: price,
        volume: Math.random() * 1000000
      });
    }

    return data;
  }

  /**
   * Generate signals for historical data
   */
  private async generateHistoricalSignals(
    _symbol: string,
    data: any[]
  ): Promise<Array<{ timestamp: number; action: 'BUY' | 'SELL' | 'NEUTRAL'; confidence: number }>> {
    const signals: Array<{ timestamp: number; action: 'BUY' | 'SELL' | 'NEUTRAL'; confidence: number }> = [];

    // Simple moving average crossover strategy for backtest
    const shortPeriod = 20;
    const longPeriod = 50;

    for (let i = longPeriod; i < data.length; i++) {
      const shortMA = this.calculateMA(data, i, shortPeriod);
      const longMA = this.calculateMA(data, i, longPeriod);
      const rsi = this.calculateRSI(data, i, 14);

      let action: 'BUY' | 'SELL' | 'NEUTRAL' = 'NEUTRAL';
      let confidence = 0.5;

      // Golden cross
      if (shortMA > longMA && rsi < 70) {
        action = 'BUY';
        confidence = 0.75;
      }

      // Death cross
      if (shortMA < longMA && rsi > 30) {
        action = 'SELL';
        confidence = 0.75;
      }

      signals.push({
        timestamp: data[i].timestamp,
        action,
        confidence
      });
    }

    return signals;
  }

  /**
   * Simulate trades based on signals
   */
  private simulateTrades(
    data: any[],
    signals: Array<{ timestamp: number; action: string; confidence: number }>
  ): BacktestTrade[] {
    const trades: BacktestTrade[] = [];
    let capital = this.INITIAL_CAPITAL;
    let position: { action: 'BUY' | 'SELL'; entryPrice: number; entryTime: number } | null = null;

    for (let i = 0; i < signals.length; i++) {
      const signal = signals[i];
      const candle = data.find(d => d.timestamp === signal.timestamp);
      if (!candle) continue;

      // Entry logic
      if (!position && signal.action !== 'NEUTRAL' && signal.confidence >= 0.70) {
        position = {
          action: signal.action as 'BUY' | 'SELL',
          entryPrice: candle.close,
          entryTime: candle.timestamp
        };
      }

      // Exit logic (after 5-10 candles or opposite signal)
      if (position) {
        const holdingPeriod = Math.floor((candle.timestamp - position.entryTime) / (1000 * 60 * 60 * 24));
        const oppositeSignal = (position.action === 'BUY' && signal.action === 'SELL') ||
                               (position.action === 'SELL' && signal.action === 'BUY');

        if (holdingPeriod >= 5 || oppositeSignal) {
          const exitPrice = candle.close;
          let profit = 0;

          if (position.action === 'BUY') {
            profit = (exitPrice - position.entryPrice) * (capital * this.POSITION_SIZE / position.entryPrice);
          } else {
            profit = (position.entryPrice - exitPrice) * (capital * this.POSITION_SIZE / position.entryPrice);
          }

          // Apply commission
          profit -= capital * this.POSITION_SIZE * this.COMMISSION * 2; // Entry + exit

          const profitPercent = (profit / (capital * this.POSITION_SIZE)) * 100;

          trades.push({
            entryTime: position.entryTime,
            exitTime: candle.timestamp,
            entryPrice: position.entryPrice,
            exitPrice,
            action: position.action,
            profit,
            profitPercent
          });

          capital += profit;
          position = null;
        }
      }
    }

    return trades;
  }

  /**
   * Calculate performance metrics
   */
  private calculateMetrics(trades: BacktestTrade[]) {
    if (trades.length === 0) {
      return {
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0,
        winRate: 0,
        totalReturn: 0,
        totalReturnPercent: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        maxDrawdownPercent: 0,
        avgProfit: 0,
        avgLoss: 0,
        profitFactor: 0
      };
    }

    const winningTrades = trades.filter(t => t.profit > 0);
    const losingTrades = trades.filter(t => t.profit < 0);

    const totalProfit = winningTrades.reduce((sum, t) => sum + t.profit, 0);
    const totalLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.profit, 0));

    const totalReturn = trades.reduce((sum, t) => sum + t.profit, 0);
    const totalReturnPercent = (totalReturn / this.INITIAL_CAPITAL) * 100;

    // Sharpe Ratio (simplified: return / std dev)
    const returns = trades.map(t => t.profitPercent);
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const stdDev = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0; // Annualized

    // Max Drawdown
    let peak = this.INITIAL_CAPITAL;
    let maxDrawdown = 0;
    let capital = this.INITIAL_CAPITAL;

    trades.forEach(trade => {
      capital += trade.profit;
      if (capital > peak) {
        peak = capital;
      }
      const drawdown = peak - capital;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    });

    const maxDrawdownPercent = (maxDrawdown / peak) * 100;

    return {
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate: (winningTrades.length / trades.length) * 100,
      totalReturn,
      totalReturnPercent,
      sharpeRatio: parseFloat(sharpeRatio.toFixed(2)),
      maxDrawdown,
      maxDrawdownPercent: parseFloat(maxDrawdownPercent.toFixed(2)),
      avgProfit: winningTrades.length > 0 ? totalProfit / winningTrades.length : 0,
      avgLoss: losingTrades.length > 0 ? totalLoss / losingTrades.length : 0,
      profitFactor: totalLoss > 0 ? totalProfit / totalLoss : 0
    };
  }

  /**
   * Calculate Simple Moving Average
   */
  private calculateMA(data: any[], index: number, period: number): number {
    const sum = data.slice(Math.max(0, index - period + 1), index + 1)
      .reduce((acc, d) => acc + d.close, 0);
    return sum / period;
  }

  /**
   * Calculate RSI
   */
  private calculateRSI(data: any[], index: number, period: number): number {
    if (index < period) return 50; // Default neutral

    let gains = 0;
    let losses = 0;

    for (let i = index - period + 1; i <= index; i++) {
      const change = data[i].close - data[i - 1].close;
      if (change > 0) {
        gains += change;
      } else {
        losses += Math.abs(change);
      }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;

    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }
}

export const backtestingEngine = new BacktestingEngine();