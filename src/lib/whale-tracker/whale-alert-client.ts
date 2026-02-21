/**
 * 🐋 WHALE ALERT API CLIENT
 *
 * Whale Alert API entegrasyonu - Gerçek blockchain whale hareketlerini izler.
 * API Docs: https://docs.whale-alert.io/
 *
 * Rate Limits:
 * - Free Tier: 10 requests/minute
 * - Personal Tier: 60 requests/minute
 *
 * WHITE-HAT: Sadece public blockchain data, read-only.
 */

import type { WhaleTransaction, Blockchain } from '@/types/whale-tracker';
import { findWalletInfo } from './wallet-database';

const WHALE_ALERT_BASE_URL = 'https://api.whale-alert.io/v1';
const WHALE_ALERT_API_KEY = process.env.WHALE_ALERT_API_KEY;

// Whale Alert API response types
interface WhaleAlertTransaction {
  blockchain: string;
  symbol: string;
  id: string;
  transaction_type: string;
  hash: string;
  from: {
    address: string;
    owner?: string;
    owner_type?: string;
  };
  to: {
    address: string;
    owner?: string;
    owner_type?: string;
  };
  timestamp: number;
  amount: number;
  amount_usd: number;
  transaction_count?: number;
}

interface WhaleAlertResponse {
  result: string;
  cursor?: string;
  count: number;
  transactions: WhaleAlertTransaction[];
}

// Map Whale Alert blockchain names to our format
function mapBlockchain(whaleAlertBlockchain: string): Blockchain | null {
  const map: Record<string, Blockchain> = {
    'bitcoin': 'BTC',
    'ethereum': 'ETH',
    'avalanche': 'AVAX'
  };

  return map[whaleAlertBlockchain.toLowerCase()] || null;
}

// Map Whale Alert owner types to our categories
function mapCategory(ownerType?: string): 'FOUNDER' | 'EXCHANGE' | 'WHALE' | 'GOVERNMENT' | 'UNKNOWN' {
  if (!ownerType) return 'UNKNOWN';

  const type = ownerType.toLowerCase();

  if (type.includes('exchange')) return 'EXCHANGE';
  if (type.includes('government') || type.includes('seized')) return 'GOVERNMENT';
  if (type === 'unknown' || type === 'wallet') return 'WHALE';

  return 'UNKNOWN';
}

// Generate Turkish description based on transaction context
function generateTurkishDescription(
  blockchain: Blockchain,
  amount: number,
  amountUSD: number,
  fromLabel?: string,
  toLabel?: string,
  fromCategory?: string,
  toCategory?: string
): string {
  // Critical: Satoshi or Genesis wallets
  if (fromLabel?.includes('Satoshi') || fromLabel?.includes('Genesis')) {
    return `🚨 KRİTİK: ${fromLabel} cüzdanından ${amount.toFixed(2)} ${blockchain} transfer edildi! Bitcoin tarihinde nadir görülen bir olay.`;
  }

  // Critical: Vitalik Buterin
  if (fromLabel?.includes('Vitalik')) {
    return `⚡ Vitalik Buterin ${amount.toFixed(2)} ${blockchain} transfer etti. ${toLabel ? `${toLabel} adresine gönderildi.` : 'Muhtemelen bir borsa veya DeFi protokolüne.'}`;
  }

  // Exchange to Exchange
  if (fromCategory === 'EXCHANGE' && toCategory === 'EXCHANGE') {
    return `🔄 ${fromLabel || 'Bir borsa'} → ${toLabel || 'başka bir borsa'}: ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(1)}M) transfer edildi.`;
  }

  // Exchange withdrawal (potential sell signal)
  if (fromCategory === 'EXCHANGE' && !toCategory?.includes('EXCHANGE')) {
    return `📤 ${fromLabel || 'Bir borsa'}'dan ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(1)}M) çekildi. Büyük müşteri çekimi olabilir.`;
  }

  // Exchange deposit (potential sell signal)
  if (!fromCategory?.includes('EXCHANGE') && toCategory === 'EXCHANGE') {
    return `📥 ${toLabel || 'Bir borsaya'} ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(1)}M) yatırıldı. Satış yapma ihtimali var.`;
  }

  // Government/Seized funds
  if (fromCategory === 'GOVERNMENT' || toCategory === 'GOVERNMENT') {
    return `⚖️ Hükümet cüzdanı hareketi: ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(1)}M). Muhtemelen el konulan varlıklar.`;
  }

  // Large whale movement
  if (amountUSD > 50_000_000) {
    return `🐋 BÜYÜK BALİNA HAREKETİ: ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(1)}M) transfer edildi. Piyasada etki yaratabilir.`;
  }

  // Medium whale movement
  if (amountUSD > 10_000_000) {
    return `💰 Önemli transfer: ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(2)}M) hareketi tespit edildi.`;
  }

  // Default
  return `📊 ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(2)}M) transfer gerçekleşti.`;
}

// Determine transaction significance
function determineSignificance(
  amountUSD: number,
  fromLabel?: string,
  fromCategory?: string
): 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' {
  // Critical: Satoshi, Genesis, or >$100M
  if (fromLabel?.includes('Satoshi') || fromLabel?.includes('Genesis') || amountUSD > 100_000_000) {
    return 'CRITICAL';
  }

  // High: Founders or >$10M
  if (fromCategory === 'FOUNDER' || amountUSD > 10_000_000) {
    return 'HIGH';
  }

  // Medium: >$1M
  if (amountUSD > 1_000_000) {
    return 'MEDIUM';
  }

  return 'LOW';
}

// Transform Whale Alert transaction to our format
function transformTransaction(whaleAlertTx: WhaleAlertTransaction): WhaleTransaction | null {
  const blockchain = mapBlockchain(whaleAlertTx.blockchain);

  // Skip unsupported blockchains
  if (!blockchain) {
    return null;
  }

  // Enrich with known wallet data
  const fromWallet = findWalletInfo(whaleAlertTx.from.address, blockchain);
  const toWallet = findWalletInfo(whaleAlertTx.to.address, blockchain);

  const fromLabel = fromWallet?.label || whaleAlertTx.from.owner;
  const toLabel = toWallet?.label || whaleAlertTx.to.owner;

  const fromCategory = fromWallet?.category || mapCategory(whaleAlertTx.from.owner_type);
  const toCategory = toWallet?.category || mapCategory(whaleAlertTx.to.owner_type);

  const turkishDescription = generateTurkishDescription(
    blockchain,
    whaleAlertTx.amount,
    whaleAlertTx.amount_usd,
    fromLabel,
    toLabel,
    fromCategory,
    toCategory
  );

  const significance = determineSignificance(
    whaleAlertTx.amount_usd,
    fromLabel,
    fromCategory
  );

  return {
    hash: whaleAlertTx.hash,
    blockchain,
    from: whaleAlertTx.from.address,
    to: whaleAlertTx.to.address,
    amount: whaleAlertTx.amount,
    amountUSD: whaleAlertTx.amount_usd,
    timestamp: whaleAlertTx.timestamp * 1000, // Convert to milliseconds

    fromLabel,
    toLabel,
    fromCategory,
    toCategory,

    turkishDescription,
    turkishFromNote: fromWallet?.turkishNote,
    turkishToNote: toWallet?.turkishNote,

    significance
  };
}

/**
 * Fetch whale transactions from Whale Alert API
 *
 * @param minValue - Minimum transaction value in USD (default: 500000)
 * @param startTime - Unix timestamp for start time (default: 10 minutes ago)
 * @param limit - Max number of transactions to return (API max: 100)
 */
export async function fetchWhaleAlertTransactions(
  minValue: number = 500000,
  startTime?: number,
  limit: number = 50
): Promise<WhaleTransaction[]> {
  if (!WHALE_ALERT_API_KEY) {
    throw new Error('Whale Alert API key not configured');
  }

  // Default to last 10 minutes if not specified
  const start = startTime || Math.floor((Date.now() - 10 * 60 * 1000) / 1000);
  const end = Math.floor(Date.now() / 1000);

  try {
    const url = new URL(`${WHALE_ALERT_BASE_URL}/transactions`);
    url.searchParams.set('api_key', WHALE_ALERT_API_KEY);
    url.searchParams.set('min_value', minValue.toString());
    url.searchParams.set('start', start.toString());
    url.searchParams.set('end', end.toString());
    url.searchParams.set('limit', Math.min(limit, 100).toString());

    console.log(`[Whale Alert] Fetching transactions: ${url.toString().replace(WHALE_ALERT_API_KEY, 'HIDDEN')}`);

    const response = await fetch(url.toString(), {
      method: 'GET',
      headers: {
        'Accept': 'application/json'
      }
    });

    if (!response.ok) {
      throw new Error(`Whale Alert API error: ${response.status} ${response.statusText}`);
    }

    const data: WhaleAlertResponse = await response.json();

    if (data.result !== 'success') {
      throw new Error(`Whale Alert API returned: ${data.result}`);
    }

    console.log(`[Whale Alert] Received ${data.count} transactions`);

    // Transform and filter transactions
    const transactions = data.transactions
      .map(transformTransaction)
      .filter((tx): tx is WhaleTransaction => tx !== null);

    console.log(`[Whale Alert] Transformed ${transactions.length} supported transactions`);

    return transactions;

  } catch (error) {
    console.error('[Whale Alert] API Error:', error);
    throw error;
  }
}

/**
 * Check if Whale Alert API is configured and available
 */
export function isWhaleAlertAvailable(): boolean {
  return !!WHALE_ALERT_API_KEY && WHALE_ALERT_API_KEY.length > 0;
}

/**
 * Test Whale Alert API connection
 */
export async function testWhaleAlertConnection(): Promise<boolean> {
  if (!isWhaleAlertAvailable()) {
    return false;
  }

  try {
    // Fetch just 1 transaction from last hour to test connection
    const oneHourAgo = Math.floor((Date.now() - 60 * 60 * 1000) / 1000);
    await fetchWhaleAlertTransactions(1000000, oneHourAgo, 1);
    return true;
  } catch (error) {
    console.error('[Whale Alert] Connection test failed:', error);
    return false;
  }
}
