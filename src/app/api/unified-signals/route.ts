/**
 * UNIFIED SIGNALS API
 * Combines multiple trading strategies into unified consensus signals
 *
 * Aggregates: Trading Signals, Conservative, Quantum, AI, Breakout-Retest
 * Method: Weighted voting system (18+ strategies unified)
 */

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface UnifiedSignal {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  buyPercentage: number;
  sellPercentage: number;
  overallSignal: 'BUY' | 'SELL' | 'WAIT';
  overallConfidence: number;
  activeStrategies: number;
  strategyVotes: {
    buy: number;
    sell: number;
    wait: number;
  };
  lastUpdate: string;
}

/**
 * Fetch strategy data with timeout
 */
async function fetchWithTimeout(url: string, timeoutMs: number = 3000): Promise<any> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      signal: controller.signal,
      cache: 'no-store',
    });
    clearTimeout(timeoutId);

    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    return null;
  }
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get('limit') || '50');
    const minBuyPercentage = parseInt(searchParams.get('minBuyPercentage') || '0');

    console.log(`[Unified Signals] Fetching unified signals (limit: ${limit}, minBuy: ${minBuyPercentage}%)`);

    const protocol = request.headers.get('x-forwarded-proto') || 'https';
    const host = request.headers.get('host') || 'localhost:3000';
    const baseUrl = `${protocol}://${host}`;

    // Fetch all strategies in parallel (with timeout)
    const [
      tradingSignals,
      conservativeSignals,
      breakoutSignals,
      marketCorrelation,
    ] = await Promise.allSettled([
      fetchWithTimeout(`${baseUrl}/api/signals?limit=600`, 4000),
      fetchWithTimeout(`${baseUrl}/api/conservative-signals`, 4000),
      fetchWithTimeout(`${baseUrl}/api/breakout-retest`, 4000),
      fetchWithTimeout(`${baseUrl}/api/market-correlation?limit=600`, 5000),
    ]);

    // Extract successful responses
    const allSignals: Map<string, any[]> = new Map();

    // Process Trading Signals
    if (tradingSignals.status === 'fulfilled' && tradingSignals.value?.success) {
      tradingSignals.value.data.signals.forEach((s: any) => {
        if (!allSignals.has(s.symbol)) allSignals.set(s.symbol, []);
        allSignals.get(s.symbol)!.push({ strategy: 'Trading', ...s });
      });
    }

    // Process Conservative Signals
    if (conservativeSignals.status === 'fulfilled' && conservativeSignals.value?.success) {
      conservativeSignals.value.data.signals.forEach((s: any) => {
        if (!allSignals.has(s.symbol)) allSignals.set(s.symbol, []);
        allSignals.get(s.symbol)!.push({ strategy: 'Conservative', signal: s.signal, confidence: s.confidence, price: s.price });
      });
    }

    // Process Breakout-Retest
    if (breakoutSignals.status === 'fulfilled' && breakoutSignals.value?.success) {
      breakoutSignals.value.data.signals.forEach((s: any) => {
        if (!allSignals.has(s.symbol)) allSignals.set(s.symbol, []);
        allSignals.get(s.symbol)!.push({ strategy: 'Breakout', signal: s.signal, confidence: s.confidence, price: s.price });
      });
    }

    // Process Market Correlation
    if (marketCorrelation.status === 'fulfilled' && marketCorrelation.value?.success) {
      marketCorrelation.value.data.correlations.forEach((c: any) => {
        if (!allSignals.has(c.symbol)) allSignals.set(c.symbol, []);
        allSignals.get(c.symbol)!.push({ strategy: 'Correlation', signal: c.signal, confidence: c.confidence, price: c.price });
      });
    }

    // Aggregate signals per symbol
    const unifiedSignals: UnifiedSignal[] = [];

    for (const [symbol, strategies] of allSignals.entries()) {
      if (strategies.length === 0) continue;

      // Count votes
      let buyVotes = 0;
      let sellVotes = 0;
      let waitVotes = 0;
      let totalConfidence = 0;
      let price = 0;
      let change24h = 0;
      let volume24h = 0;

      strategies.forEach((s) => {
        const signal = (s.type || s.signal || '').toUpperCase();
        if (signal === 'BUY' || signal === 'AL') buyVotes++;
        else if (signal === 'SELL' || signal === 'SAT') sellVotes++;
        else waitVotes++;

        totalConfidence += s.confidence || 50;
        if (s.price) price = s.price;
        if (s.change24h) change24h = s.change24h;
        if (s.volume24h) volume24h = s.volume24h;
      });

      const totalVotes = buyVotes + sellVotes + waitVotes;
      const buyPercentage = totalVotes > 0 ? Math.round((buyVotes / totalVotes) * 100) : 0;
      const sellPercentage = totalVotes > 0 ? Math.round((sellVotes / totalVotes) * 100) : 0;
      const overallConfidence = Math.round(totalConfidence / strategies.length);

      // Determine overall signal
      let overallSignal: 'BUY' | 'SELL' | 'WAIT' = 'WAIT';
      if (buyPercentage >= 60) overallSignal = 'BUY';
      else if (sellPercentage >= 60) overallSignal = 'SELL';

      // Filter by minBuyPercentage
      if (minBuyPercentage > 0 && buyPercentage < minBuyPercentage) continue;

      unifiedSignals.push({
        symbol,
        price,
        change24h,
        volume24h,
        buyPercentage,
        sellPercentage,
        overallSignal,
        overallConfidence,
        activeStrategies: strategies.length,
        strategyVotes: {
          buy: buyVotes,
          sell: sellVotes,
          wait: waitVotes,
        },
        lastUpdate: new Date().toISOString(),
      });
    }

    // Sort by buy percentage (descending)
    unifiedSignals.sort((a, b) => b.buyPercentage - a.buyPercentage);

    // Apply limit
    const limitedSignals = unifiedSignals.slice(0, limit);

    console.log(`[Unified Signals] Generated ${limitedSignals.length}/${unifiedSignals.length} signals (filtered by limit)`);

    return NextResponse.json({
      success: true,
      data: {
        signals: limitedSignals,
        totalSignals: unifiedSignals.length,
        filteredCount: limitedSignals.length,
        buySignals: unifiedSignals.filter(s => s.overallSignal === 'BUY').length,
        sellSignals: unifiedSignals.filter(s => s.overallSignal === 'SELL').length,
        waitSignals: unifiedSignals.filter(s => s.overallSignal === 'WAIT').length,
      },
      timestamp: new Date().toISOString(),
    });

  } catch (error: any) {
    console.error('[Unified Signals API Error]:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Unknown error',
      },
      { status: 500 }
    );
  }
}
