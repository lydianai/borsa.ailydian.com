/**
 * 🐋 WHALE TRANSACTION TRACKER TYPES
 *
 * Büyük Bitcoin, Ethereum ve Avalanche cüzdan hareketlerini
 * izlemek için type definitions.
 *
 * WHITE-HAT PRINCIPLES:
 * - Sadece public blockchain data
 * - Read-only, hiçbir cüzdana müdahale yok
 * - Educational purpose only
 * - Privacy respecting
 */

export type Blockchain = 'BTC' | 'ETH' | 'AVAX';

export type WalletCategory = 'FOUNDER' | 'EXCHANGE' | 'WHALE' | 'GOVERNMENT' | 'UNKNOWN';

export type TransactionSignificance = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface KnownWallet {
  address: string;
  label: string;
  owner?: string;
  category: WalletCategory;
  description: string;
  balance?: number;
  firstSeen?: string;
  turkishNote?: string;
}

export interface WhaleTransaction {
  hash: string;
  blockchain: Blockchain;
  from: string;
  to: string;
  amount: number;
  amountUSD: number;
  timestamp: number;

  // Enriched metadata
  fromLabel?: string;
  toLabel?: string;
  fromCategory?: WalletCategory;
  toCategory?: WalletCategory;

  // Turkish descriptions
  turkishDescription: string;
  turkishFromNote?: string;
  turkishToNote?: string;

  // Significance
  significance: TransactionSignificance;

  // Additional data
  blockNumber?: number;
  confirmations?: number;
}

export interface WhaleStats {
  last24h: {
    transactionCount: number;
    totalVolumeUSD: number;
    largestTransaction: {
      amount: number;
      blockchain: Blockchain;
      amountUSD: number;
    };
  };
  byBlockchain: {
    [key in Blockchain]: {
      count: number;
      volumeUSD: number;
    };
  };
}

export interface WhaleTrackerFilters {
  blockchain?: Blockchain | 'ALL';
  minThresholdUSD?: number;
  significance?: TransactionSignificance[];
  category?: WalletCategory[];
  timeRange?: '1h' | '24h' | '7d' | '30d';
}

export interface WhaleTrackerAPIResponse {
  success: boolean;
  data?: {
    transactions: WhaleTransaction[];
    stats: WhaleStats;
    lastUpdate: number;
  };
  error?: string;
  metadata?: {
    duration: number;
    timestamp: number;
    cached?: boolean;
  };
}
