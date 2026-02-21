/**
 * 🤖 BOT ANALYSIS TYPES
 *
 * Binance Futures perpetual market bot behavior analysis types
 *
 * WHITE-HAT PRINCIPLES:
 * - Read-only market data analysis
 * - Educational and research purposes only
 * - No automated trading execution
 * - Transparent data collection
 * - Rate-limited API calls
 */

// ============================================================================
// ORDER BOOK TYPES
// ============================================================================

export interface OrderBookLevel {
  price: number;
  quantity: number;
  total: number; // Cumulative quantity
  usdValue: number; // Price * Quantity in USD
}

export interface OrderBookSnapshot {
  symbol: string;
  timestamp: number;
  bids: OrderBookLevel[]; // Buy orders
  asks: OrderBookLevel[]; // Sell orders
  bidVolume: number; // Total bid volume
  askVolume: number; // Total ask volume
  spread: number; // Ask price - Bid price
  spreadPercent: number; // (Spread / Mid price) * 100
  midPrice: number; // (Best bid + Best ask) / 2
}

export interface OrderBookImbalance {
  symbol: string;
  timestamp: number;
  ratio: number; // Bid volume / Ask volume
  signal: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  confidence: number; // 0-100
  bidPressure: number; // 0-100
  askPressure: number; // 0-100
}

export interface WhaleLevelDetection {
  price: number;
  side: 'BID' | 'ASK';
  quantity: number;
  usdValue: number;
  isWhaleWall: boolean; // > $500K
  percentageOfDepth: number; // % of total orderbook depth
}

// ============================================================================
// FUNDING RATE TYPES
// ============================================================================

export interface FundingRate {
  symbol: string;
  fundingRate: number; // Actual funding rate (e.g., 0.0001 = 0.01%)
  fundingTime: number; // Next funding timestamp
  markPrice: number;
  indexPrice: number;
  estimatedSettlePrice: number;
  lastFundingRate: number;
  nextFundingTime: number;
  interestRate: number;
  time: number;
}

export interface FundingRateHistory {
  symbol: string;
  data: Array<{
    fundingRate: number;
    fundingTime: number;
    time: number;
  }>;
  avgFundingRate: number; // Average over period
  minFundingRate: number;
  maxFundingRate: number;
  trend: 'INCREASING' | 'DECREASING' | 'STABLE';
}

export interface FundingRateSignal {
  symbol: string;
  currentRate: number;
  signal: 'LONG_OPPORTUNITY' | 'SHORT_OPPORTUNITY' | 'NEUTRAL' | 'EXTREME_LONG' | 'EXTREME_SHORT';
  confidence: number; // 0-100
  reason: string;
  recommendation: string;
}

// ============================================================================
// OPEN INTEREST & VOLUME TYPES
// ============================================================================

export interface OpenInterestData {
  symbol: string;
  sumOpenInterest: string; // Total open interest
  sumOpenInterestValue: string; // In USD
  timestamp: number;
}

export interface OpenInterestChange {
  symbol: string;
  current: number;
  previous: number;
  changePercent: number;
  changeAbsolute: number;
  priceChange: number;
  volumeChange: number;
  signal: 'ACCUMULATION' | 'DISTRIBUTION' | 'NEUTRAL';
  interpretation: string;
}

export interface CVDData {
  symbol: string;
  timestamp: number;
  cumulativeVolumeDelta: number; // Cumulative buyer volume - seller volume
  buyVolume: number;
  sellVolume: number;
  netVolume: number;
  trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
}

// ============================================================================
// LIQUIDATION TYPES
// ============================================================================

export interface LiquidationEvent {
  symbol: string;
  side: 'BUY' | 'SELL'; // Side of liquidation
  price: number;
  quantity: number;
  time: number;
  isLargeOrder: boolean; // > $100K
}

export interface LiquidationCluster {
  priceLevel: number;
  totalQuantity: number;
  totalUsdValue: number;
  eventCount: number;
  side: 'LONG' | 'SHORT' | 'MIXED';
  density: number; // Events per price range
}

export interface LiquidationHeatmap {
  symbol: string;
  timestamp: number;
  clusters: LiquidationCluster[];
  totalLiquidations: number;
  longLiquidations: number;
  shortLiquidations: number;
  dominantSide: 'LONG' | 'SHORT' | 'BALANCED';
  nearestCluster: LiquidationCluster | null; // Closest to current price
}

// ============================================================================
// BOT BEHAVIOR PATTERN TYPES
// ============================================================================

export interface BotBehaviorPattern {
  type: 'MARKET_MAKING' | 'ARBITRAGE' | 'TREND_FOLLOWING' | 'MEAN_REVERSION';
  confidence: number; // 0-100
  detectedAt: number;
  indicators: string[];
  characteristics: {
    orderFrequency?: number; // Orders per minute
    avgOrderSize?: number;
    spreadUtilization?: number; // For market makers
    responseTime?: number; // Time to react to price changes (ms)
  };
}

export interface MarketMicrostructure {
  symbol: string;
  timestamp: number;
  orderBookImbalance: OrderBookImbalance;
  fundingRateSignal: FundingRateSignal;
  liquidationHeatmap: LiquidationHeatmap;
  openInterestChange: OpenInterestChange;
  cvd: CVDData;
  detectedBotPatterns: BotBehaviorPattern[];
}

// ============================================================================
// LONG POSITION SIGNAL TYPES
// ============================================================================

export interface LongPositionSignal {
  symbol: string;
  timestamp: number;
  overallScore: number; // 0-100 composite score
  quality: 'EXCELLENT' | 'GOOD' | 'MODERATE' | 'POOR' | 'NONE';
  shouldNotify: boolean;

  // Component scores
  scores: {
    orderbook: number; // 0-100
    funding: number; // 0-100
    liquidation: number; // 0-100
    openInterest: number; // 0-100
    whale: number; // 0-100
    technical: number; // 0-100
  };

  // Entry/Exit recommendations
  entry: {
    recommendedPrice: number;
    priceRange: { min: number; max: number };
    positionSize: number; // Recommended size in USD
    leverage: number; // Recommended leverage
  };

  exit: {
    stopLoss: number;
    takeProfits: Array<{ price: number; percentage: number }>; // TP levels
    trailingStop: number;
  };

  riskReward: {
    ratio: number; // e.g., 3.0 means 3:1 reward:risk
    potentialGain: number; // %
    potentialLoss: number; // %
    confidence: number; // 0-100
  };

  summary: string;
  reasons: string[];
  warnings: string[];
}

// ============================================================================
// API RESPONSE TYPES
// ============================================================================

export interface BotAnalysisAPIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: {
    duration: number;
    timestamp: number;
    cached?: boolean;
  };
}
