/**
 * TELEGRAM LIVE NOTIFICATION API
 * Manual trigger for Telegram notifications
 */

import { NextResponse } from 'next/server';
import { realtimeTelegramNotifier } from '@/lib/telegram/realtime-notifier';
import { indicatorsAnalyzer } from '@/lib/indicators/analyzer';
import liveFeedManager from '@/lib/data/live-feed';

// ============================================================================
// API ROUTE HANDLERS
// ============================================================================

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { symbol, forceSend } = body;
    
    if (!symbol) {
      return NextResponse.json({
        success: false,
        error: 'Symbol is required'
      }, { status: 400 });
    }
    
    // Get latest price data
    const priceData = liveFeedManager.getPrice(symbol);
    if (!priceData) {
      return NextResponse.json({
        success: false,
        error: 'Price data not available for symbol'
      }, { status: 404 });
    }
    
    // Perform technical analysis
    const analysis = await indicatorsAnalyzer.analyze(symbol, priceData);
    
    if (forceSend) {
      // Force send notification
      await realtimeTelegramNotifier.forceSend(symbol, analysis);
      return NextResponse.json({
        success: true,
        message: 'Force notification sent',
        symbol,
        analysis
      });
    } else {
      // Handle live signal normally
      await realtimeTelegramNotifier.handleLiveSignal(symbol, analysis);
      return NextResponse.json({
        success: true,
        message: 'Signal processed',
        symbol,
        analysis
      });
    }
  } catch (error: any) {
    console.error('[TelegramLiveAPI] POST error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}