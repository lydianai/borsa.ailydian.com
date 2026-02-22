/**
 * 📊 LyTrade TRADING STRATEGIES - Type Definitions
 * 13 Strategy Modules for Real-Time Analysis
 */

export interface Candle {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}

export interface PriceData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  high24h: number;
  low24h: number;
  candles?: Candle[]; // Optional historical candles
}

export type SignalType = 'BUY' | 'SELL' | 'WAIT' | 'NEUTRAL';

export interface StrategySignal {
  name: string;
  signal: SignalType;
  confidence: number; // 0-100
  reason: string;
  targets?: number[];
  stopLoss?: number;
  timeframe?: string;
  indicators?: Record<string, number | string>;
}

export interface StrategyAnalysis {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  groqAnalysis?: string;
  strategies: StrategySignal[];
  overallScore: number; // 0-100
  recommendation: 'STRONG_BUY' | 'BUY' | 'NEUTRAL' | 'WAIT' | 'SELL' | 'STRONG_SELL';
  buyCount: number;
  waitCount: number;
  sellCount: number;
  neutralCount: number;
  timestamp: string;
}

export interface StrategyModule {
  name: string;
  analyze: (data: PriceData) => Promise<StrategySignal>;
}
