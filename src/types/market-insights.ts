/**
 * MARKET INSIGHTS TYPE DEFINITIONS
 * TypeScript interfaces for Market Insights data
 */

export interface FundingRateData {
  symbol: string;
  funding_rate: number;
  funding_time: number;
  timestamp: string;
}

export interface OpenInterestData {
  symbol: string;
  open_interest: number;
  timestamp: string;
  time_ms: number;
}

export interface LiquidationHeatmapPoint {
  price: number;
  liquidation_amount_usd: number;
  timestamp: string;
}

export interface LiquidationHeatmapData {
  symbol: string;
  current_price: number;
  heatmap: LiquidationHeatmapPoint[];
}

export interface LongShortRatioData {
  symbol: string;
  long_percentage: number;
  short_percentage: number;
  long_account: number;
  short_account: number;
  timestamp: string;
}

export interface MarketData {
  symbol: string;
  price: number;
  price_change: number;
  price_change_percent: number;
  high_price: number;
  low_price: number;
  volume: number;
  quote_volume: number;
}

export interface PremiumIndexData {
  symbol: string;
  mark_price: number;
  index_price: number;
  last_funding_rate: number;
  next_funding_time: number;
  next_funding_timestamp: string;
}
