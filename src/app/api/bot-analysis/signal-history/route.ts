/**
 * 📊 SIGNAL HISTORY TRACKER API
 *
 * Tracks and stores LONG signal scores over time for trend analysis
 * - Stores signal scores every 30 seconds (when auto-refresh runs)
 * - Returns last 24 hours of signal history
 * - Provides trend analysis (improving/declining)
 *
 * WHITE-HAT PRINCIPLES:
 * - In-memory storage only (no persistent DB)
 * - Educational purpose only
 * - Read-only analysis
 *
 * USAGE:
 * POST /api/bot-analysis/signal-history - Add new signal score
 * GET /api/bot-analysis/signal-history?symbol=BTCUSDT - Get history for symbol
 */

import { NextRequest, NextResponse } from 'next/server';
import type { BotAnalysisAPIResponse } from '@/types/bot-analysis';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// ============================================================================
// TYPES
// ============================================================================

interface SignalHistoryEntry {
  timestamp: number;
  score: number;
  quality: 'EXCELLENT' | 'GOOD' | 'MODERATE' | 'POOR' | 'NONE';
}

interface SignalHistory {
  symbol: string;
  entries: SignalHistoryEntry[];
  trend: 'IMPROVING' | 'DECLINING' | 'STABLE';
  avgScore24h: number;
  currentScore: number;
  scoreChange24h: number;
}

// ============================================================================
// IN-MEMORY STORAGE
// ============================================================================

// Store signal history in memory (resets on server restart)
// Key: symbol, Value: array of entries
const signalHistoryStore = new Map<string, SignalHistoryEntry[]>();

const MAX_HISTORY_HOURS = 24;
const MAX_ENTRIES_PER_SYMBOL = (MAX_HISTORY_HOURS * 60 * 60) / 30; // 30-second intervals = 2880 max entries

// ============================================================================
// HELPER: CLEAN OLD ENTRIES
// ============================================================================

function cleanOldEntries(entries: SignalHistoryEntry[]): SignalHistoryEntry[] {
  const now = Date.now();
  const cutoff = now - (MAX_HISTORY_HOURS * 60 * 60 * 1000); // 24 hours ago

  return entries.filter(entry => entry.timestamp >= cutoff);
}

// ============================================================================
// HELPER: CALCULATE TREND
// ============================================================================

function calculateTrend(entries: SignalHistoryEntry[]): 'IMPROVING' | 'DECLINING' | 'STABLE' {
  if (entries.length < 10) return 'STABLE'; // Not enough data

  // Compare recent average (last 10 entries) vs older average (10-20 entries ago)
  const recent = entries.slice(-10);
  const older = entries.slice(-20, -10);

  if (older.length === 0) return 'STABLE';

  const recentAvg = recent.reduce((sum, e) => sum + e.score, 0) / recent.length;
  const olderAvg = older.reduce((sum, e) => sum + e.score, 0) / older.length;

  const diff = recentAvg - olderAvg;

  if (diff > 5) return 'IMPROVING';
  if (diff < -5) return 'DECLINING';
  return 'STABLE';
}

// ============================================================================
// POST HANDLER - ADD NEW SIGNAL SCORE
// ============================================================================

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { symbol, score, quality } = body;

    if (!symbol || score === undefined || !quality) {
      return NextResponse.json(
        {
          success: false,
          error: 'Missing required fields: symbol, score, quality'
        },
        { status: 400 }
      );
    }

    // Get existing entries or create new array
    let entries = signalHistoryStore.get(symbol) || [];

    // Add new entry
    const newEntry: SignalHistoryEntry = {
      timestamp: Date.now(),
      score,
      quality
    };

    entries.push(newEntry);

    // Clean old entries (keep only last 24 hours)
    entries = cleanOldEntries(entries);

    // Limit to max entries to prevent memory overflow
    if (entries.length > MAX_ENTRIES_PER_SYMBOL) {
      entries = entries.slice(-MAX_ENTRIES_PER_SYMBOL);
    }

    // Update store
    signalHistoryStore.set(symbol, entries);

    console.log(
      `[Signal History] Added entry for ${symbol}: score=${score}, quality=${quality}, total_entries=${entries.length}`
    );

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        entriesCount: entries.length,
        latestScore: score
      }
    });

  } catch (error) {
    console.error('[Signal History POST] Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to add signal history'
      },
      { status: 500 }
    );
  }
}

// ============================================================================
// GET HANDLER - RETRIEVE SIGNAL HISTORY
// ============================================================================

export async function GET(request: NextRequest) {
  const startTime = Date.now();
  const { searchParams } = request.nextUrl;

  const symbol = searchParams.get('symbol') || 'BTCUSDT';
  const limit = parseInt(searchParams.get('limit') || '100');

  try {
    console.log(`[Signal History GET] Fetching history for ${symbol}...`);

    // Get entries for symbol
    let entries = signalHistoryStore.get(symbol) || [];

    // Clean old entries
    entries = cleanOldEntries(entries);

    if (entries.length === 0) {
      // No history yet
      const response: BotAnalysisAPIResponse<SignalHistory> = {
        success: true,
        data: {
          symbol,
          entries: [],
          trend: 'STABLE',
          avgScore24h: 0,
          currentScore: 0,
          scoreChange24h: 0
        },
        metadata: {
          duration: Date.now() - startTime,
          timestamp: Date.now()
        }
      };

      return NextResponse.json(response);
    }

    // Calculate statistics
    const avgScore24h = entries.reduce((sum, e) => sum + e.score, 0) / entries.length;
    const currentScore = entries[entries.length - 1].score;
    const firstScore = entries[0].score;
    const scoreChange24h = currentScore - firstScore;

    // Calculate trend
    const trend = calculateTrend(entries);

    // Limit entries to requested limit
    const limitedEntries = entries.slice(-limit);

    const duration = Date.now() - startTime;

    console.log(
      `[Signal History GET] ${symbol}: ${entries.length} entries, avg=${avgScore24h.toFixed(1)}, trend=${trend}, duration=${duration}ms`
    );

    const response: BotAnalysisAPIResponse<SignalHistory> = {
      success: true,
      data: {
        symbol,
        entries: limitedEntries,
        trend,
        avgScore24h,
        currentScore,
        scoreChange24h
      },
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(response);

  } catch (error) {
    const duration = Date.now() - startTime;
    console.error('[Signal History GET] Error:', error);

    const errorResponse: BotAnalysisAPIResponse<never> = {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to fetch signal history',
      metadata: {
        duration,
        timestamp: Date.now()
      }
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}
