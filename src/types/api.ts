// src/types/api.ts

/**
 * Base API Response with Mock Data Warning
 */
export interface BaseAPIResponse {
  success: boolean;
  isMockData?: boolean; // White-hat: Clearly indicates when data is mock/demo
  mockDataWarning?: string; // Optional warning message
  timestamp?: string;
}

export interface BinanceFuturesSymbol {
  symbol: string;
  pair: string;
  contractType: string;
  status: string;
  baseAsset: string;
  quoteAsset: string;
  pricePrecision: number;
  quantityPrecision: number;
}

export interface BinanceFuturesTicker {
  symbol: string;
  price: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  quoteVolume: string;
  openTime: number;
  closeTime: number;
  count: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
  lastUpdate: string;
}

export interface BinanceData {
  success: boolean;
  data: {
    all: MarketData[];
    topVolume: MarketData[];
    topGainers: MarketData[];
    totalMarkets: number;
    lastUpdate: string;
  };
}

export interface TradingSignal {
  id: string;
  symbol: string;
  type: "BUY" | "SELL" | "HOLD";
  price: number;
  confidence: number;
  strength: number; // 1-10
  strategy: string;
  targets?: string[];
  timestamp: string;
}

export interface AISignal {
  id: string;
  symbol: string;
  type: "BUY" | "SELL" | "HOLD";
  price: number;
  confidence: number;
  strength: number;
  strategy: string;
  reasoning: string;
  targets?: string[];
  timestamp: string;
  aiModel: string;
}

export interface QuantumSignal {
  id: string;
  symbol: string;
  type: "BUY" | "SELL" | "HOLD";
  price: number;
  confidence: number;
  strength: number;
  strategy: string;
  reasoning: string;
  targets?: string[];
  timestamp: string;
  quantumScore: number;
  quantumAdvantage: number;
  portfolioOptimization?: {
    optimalWeight: number;
    expectedReturn: number;
    risk: number;
    sharpeRatio: number;
  };
  riskAnalysis?: {
    valueAtRisk: number;
    conditionalVaR: number;
    expectedShortfall: number;
    quantumSpeedup: number;
  };
}