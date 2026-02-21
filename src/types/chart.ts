/**
 * CHART TYPES - TradingView Lightweight Charts Integration
 * Professional crypto chart with support/resistance levels
 */

import { Time, ISeriesApi, IChartApi } from 'lightweight-charts';

// Binance Kline Data (OHLCV)
export interface KlineData {
  time: Time;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// Support/Resistance Level
export interface SupportResistanceLevel {
  price: number;
  strength: number; // 1-10
  type: 'support' | 'resistance';
  touches: number; // How many times price touched this level
}

// Chart Timeframe
export type Timeframe = '1m' | '5m' | '15m' | '30m' | '1h' | '2h' | '4h' | '1d';

// Timeframe to Binance API interval mapping
export const TIMEFRAME_MAP: Record<Timeframe, string> = {
  '1m': '1m',
  '5m': '5m',
  '15m': '15m',
  '30m': '30m',
  '1h': '1h',
  '2h': '2h',
  '4h': '4h',
  '1d': '1d',
};

// Chart Configuration
export interface ChartConfig {
  symbol: string;
  timeframe: Timeframe;
  showVolume: boolean;
  showSupportResistance: boolean;
  showIndicators: {
    rsi: boolean;
    macd: boolean;
    bollingerBands: boolean;
  };
}

// API Response Types
export interface ChartDataResponse {
  success: boolean;
  data: {
    klines: KlineData[];
    supportResistance?: SupportResistanceLevel[];
  };
  error?: string;
}

// Indicator Data
export interface RSIData {
  time: Time;
  value: number;
}

export interface MACDData {
  time: Time;
  macd: number;
  signal: number;
  histogram: number;
}

export interface BollingerBandsData {
  time: Time;
  upper: number;
  middle: number;
  lower: number;
}

// Chart Instance Refs
export interface ChartRefs {
  chart: IChartApi | null;
  candlestickSeries: ISeriesApi<'Candlestick'> | null;
  volumeSeries: ISeriesApi<'Histogram'> | null;
}
