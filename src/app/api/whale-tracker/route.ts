/**
 * 🐋 WHALE TRANSACTION TRACKER API
 *
 * Büyük Bitcoin, Ethereum ve Avalanche transferlerini izler.
 * Satoshi Nakamoto, Vitalik Buterin ve diğer whale'lerin hareketleri.
 *
 * WHITE-HAT: Sadece public blockchain data, read-only.
 *
 * USAGE: GET /api/whale-tracker?blockchain=BTC&limit=20
 */

import { NextRequest, NextResponse } from 'next/server';
import { findWalletInfo, KNOWN_WALLETS } from '@/lib/whale-tracker/wallet-database';
import { fetchWhaleAlertTransactions, isWhaleAlertAvailable } from '@/lib/whale-tracker/whale-alert-client';
import type { WhaleTransaction, WhaleStats, Blockchain } from '@/types/whale-tracker';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// Mock data generator (gerçek API entegrasyonu sonra eklenebilir)
function generateMockWhaleTransactions(count: number = 20): WhaleTransaction[] {
  const transactions: WhaleTransaction[] = [];
  const blockchains: Blockchain[] = ['BTC', 'ETH', 'AVAX'];

  const now = Date.now();

  for (let i = 0; i < count; i++) {
    const blockchain = blockchains[Math.floor(Math.random() * blockchains.length)];
    const knownWallets = KNOWN_WALLETS[blockchain];

    // Randomly pick from or to known wallets
    const useKnownFrom = Math.random() > 0.5;
    const useKnownTo = Math.random() > 0.3;

    const fromWallet = useKnownFrom && knownWallets.length > 0
      ? knownWallets[Math.floor(Math.random() * knownWallets.length)]
      : null;

    const toWallet = useKnownTo && knownWallets.length > 0
      ? knownWallets[Math.floor(Math.random() * knownWallets.length)]
      : null;

    // Generate amounts based on blockchain
    let amount = 0;
    let pricePerUnit = 0;

    if (blockchain === 'BTC') {
      amount = Math.random() * 500 + 50; // 50-550 BTC
      pricePerUnit = 95000;
    } else if (blockchain === 'ETH') {
      amount = Math.random() * 5000 + 500; // 500-5500 ETH
      pricePerUnit = 3500;
    } else {
      amount = Math.random() * 50000 + 5000; // 5000-55000 AVAX
      pricePerUnit = 40;
    }

    const amountUSD = amount * pricePerUnit;

    // Generate Turkish description
    let turkishDescription = '';

    if (fromWallet?.label.includes('Satoshi')) {
      turkishDescription = `🚨 KRİTİK: Satoshi Nakamoto'nun cüzdanından ${amount.toFixed(2)} BTC transfer edildi! Bu Bitcoin tarihinde çok nadir görülen bir olay.`;
    } else if (fromWallet?.label.includes('Vitalik')) {
      turkishDescription = `⚡ Vitalik Buterin ${amount.toFixed(2)} ETH transfer etti. Muhtemelen bir borsa veya DeFi protokolüne gönderdi.`;
    } else if (fromWallet?.category === 'EXCHANGE') {
      if (toWallet) {
        turkishDescription = `📊 ${fromWallet.label} borsasından ${toWallet.label} adresine ${amount.toFixed(2)} ${blockchain} transfer edildi.`;
      } else {
        turkishDescription = `📤 ${fromWallet.label} borsasından bilinmeyen bir cüzdana ${amount.toFixed(2)} ${blockchain} çekildi. Büyük bir müşteri çekim işlemi olabilir.`;
      }
    } else if (toWallet?.category === 'EXCHANGE') {
      turkishDescription = `📥 Bilinmeyen bir whale ${toWallet.label} borsasına ${amount.toFixed(2)} ${blockchain} yatırdı. Satış yapma ihtimali var.`;
    } else if (amountUSD > 50_000_000) {
      turkishDescription = `🐋 BÜYÜK BALİNA HAREKETİ: ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(1)}M) transfer edildi. Piyasada etki yaratabilir.`;
    } else {
      turkishDescription = `💰 ${amount.toFixed(2)} ${blockchain} ($${(amountUSD / 1_000_000).toFixed(2)}M) büyük bir transfer gerçekleşti.`;
    }

    // Determine significance
    let significance: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW' = 'LOW';

    if (fromWallet?.label.includes('Satoshi') || fromWallet?.label.includes('Genesis')) {
      significance = 'CRITICAL';
    } else if (amountUSD > 100_000_000) {
      significance = 'CRITICAL';
    } else if (amountUSD > 10_000_000 || fromWallet?.category === 'FOUNDER') {
      significance = 'HIGH';
    } else if (amountUSD > 1_000_000) {
      significance = 'MEDIUM';
    }

    transactions.push({
      hash: `0x${Math.random().toString(16).substring(2, 66)}`,
      blockchain,
      from: fromWallet?.address || `0x${Math.random().toString(16).substring(2, 42)}`,
      to: toWallet?.address || `0x${Math.random().toString(16).substring(2, 42)}`,
      amount,
      amountUSD,
      timestamp: now - (i * 60000 * Math.random() * 120), // Last 2 hours

      fromLabel: fromWallet?.label,
      toLabel: toWallet?.label,
      fromCategory: fromWallet?.category,
      toCategory: toWallet?.category,

      turkishDescription,
      turkishFromNote: fromWallet?.turkishNote,
      turkishToNote: toWallet?.turkishNote,

      significance
    });
  }

  return transactions.sort((a, b) => b.timestamp - a.timestamp);
}

// Calculate stats
function calculateStats(transactions: WhaleTransaction[]): WhaleStats {
  const last24h = Date.now() - 24 * 60 * 60 * 1000;
  const recent = transactions.filter(tx => tx.timestamp > last24h);

  let totalVolume = 0;
  let largest = { amount: 0, blockchain: 'BTC' as Blockchain, amountUSD: 0 };

  const byBlockchain: WhaleStats['byBlockchain'] = {
    BTC: { count: 0, volumeUSD: 0 },
    ETH: { count: 0, volumeUSD: 0 },
    AVAX: { count: 0, volumeUSD: 0 }
  };

  for (const tx of recent) {
    totalVolume += tx.amountUSD;
    byBlockchain[tx.blockchain].count++;
    byBlockchain[tx.blockchain].volumeUSD += tx.amountUSD;

    if (tx.amountUSD > largest.amountUSD) {
      largest = {
        amount: tx.amount,
        blockchain: tx.blockchain,
        amountUSD: tx.amountUSD
      };
    }
  }

  return {
    last24h: {
      transactionCount: recent.length,
      totalVolumeUSD: totalVolume,
      largestTransaction: largest
    },
    byBlockchain
  };
}

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const blockchain = (searchParams.get('blockchain') || 'ALL') as Blockchain | 'ALL';
  const limit = parseInt(searchParams.get('limit') || '20');

  try {
    let transactions: WhaleTransaction[] = [];
    let usingRealAPI = false;

    // Try to use Whale Alert API if available
    if (isWhaleAlertAvailable()) {
      console.log(`[Whale Tracker] Using Whale Alert API for real-time data...`);

      try {
        // Fetch transactions from last 10 minutes with min $500k value
        const realTransactions = await fetchWhaleAlertTransactions(500000, undefined, 50);

        if (realTransactions.length > 0) {
          transactions = realTransactions;
          usingRealAPI = true;
          console.log(`[Whale Tracker] ✅ Fetched ${transactions.length} real transactions from Whale Alert`);
        } else {
          console.log(`[Whale Tracker] ⚠️ No recent whale transactions, falling back to mock data`);
          transactions = generateMockWhaleTransactions(50);
        }
      } catch (apiError) {
        console.error('[Whale Tracker] Whale Alert API error, falling back to mock data:', apiError);
        transactions = generateMockWhaleTransactions(50);
      }
    } else {
      console.log(`[Whale Tracker] Whale Alert API not configured, using mock data`);
      transactions = generateMockWhaleTransactions(50);
    }

    // Filter by blockchain if specified
    if (blockchain !== 'ALL') {
      transactions = transactions.filter(tx => tx.blockchain === blockchain);
    }

    // Limit results
    transactions = transactions.slice(0, limit);

    // Calculate stats
    const stats = calculateStats(transactions);

    const duration = Date.now() - startTime;

    console.log(`[Whale Tracker] Returned ${transactions.length} transactions in ${duration}ms (${usingRealAPI ? 'REAL API' : 'MOCK DATA'})`);

    return NextResponse.json({
      success: true,
      data: {
        transactions,
        stats,
        lastUpdate: Date.now()
      },
      metadata: {
        duration,
        timestamp: Date.now(),
        cached: false,
        source: usingRealAPI ? 'whale-alert-api' : 'mock-data'
      }
    });

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Whale Tracker] Error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Whale tracker failed',
        metadata: {
          duration,
          timestamp: Date.now()
        }
      },
      { status: 500 }
    );
  }
}
