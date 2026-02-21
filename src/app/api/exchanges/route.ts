/**
 * Exchange API Management - List and Add
 *
 * White-hat compliance: User manages their own exchange API keys
 * GET: List user's connected exchanges
 * POST: Add new exchange API connection
 */

import { NextRequest, NextResponse } from 'next/server';
import { getCurrentUser, requireAuth, requirePayment } from '@/lib/auth/helpers';
import { prisma } from '@/lib/prisma';
import { addExchangeAPI, testExchangeConnection } from '@/lib/exchanges/manager';

/**
 * GET /api/exchanges
 * List all exchange APIs for the current user
 */
export async function GET(_req: NextRequest) {
  try {
    const user = await requirePayment();

    const exchanges = await prisma.exchangeAPI.findMany({
      where: {
        userId: user.id,
      },
      select: {
        id: true,
        exchange: true,
        name: true,
        isActive: true,
        isConnected: true,
        lastTestedAt: true,
        lastBalanceCheck: true,
        connectionError: true,
        hasWithdrawPerm: true,
        hasSpotTrading: true,
        hasFuturesTrading: true,
        permissions: true,
        disclaimerAccepted: true,
        termsAcceptedAt: true,
        testnet: true,
        createdAt: true,
        updatedAt: true,
        // NEVER return encrypted credentials in list
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    return NextResponse.json({
      success: true,
      exchanges,
      total: exchanges.length,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch exchanges' },
      { status: error.message === 'Payment required' ? 402 : 500 }
    );
  }
}

/**
 * POST /api/exchanges
 * Add new exchange API connection
 */
export async function POST(req: NextRequest) {
  try {
    const user = await requirePayment();

    const body = await req.json();
    const {
      exchange,
      name,
      apiKey,
      apiSecret,
      passphrase,
      termsAccepted,
      disclaimerAccepted,
      testnet,
    } = body;

    // Validation
    if (!exchange || !name || !apiKey || !apiSecret) {
      return NextResponse.json(
        { error: 'Missing required fields: exchange, name, apiKey, apiSecret' },
        { status: 400 }
      );
    }

    if (!termsAccepted || !disclaimerAccepted) {
      return NextResponse.json(
        { error: 'You must accept the terms and disclaimer' },
        { status: 400 }
      );
    }

    // Validate exchange type
    const validExchanges = ['okx', 'bybit', 'coinbase', 'kraken', 'btcturk'];
    if (!validExchanges.includes(exchange)) {
      return NextResponse.json(
        { error: `Invalid exchange. Supported: ${validExchanges.join(', ')}` },
        { status: 400 }
      );
    }

    // OKX requires passphrase
    if (exchange === 'okx' && !passphrase) {
      return NextResponse.json(
        { error: 'OKX requires a passphrase' },
        { status: 400 }
      );
    }

    // Check if user already has this exchange connected with same name
    const existing = await prisma.exchangeAPI.findFirst({
      where: {
        userId: user.id,
        exchange,
        name,
      },
    });

    if (existing) {
      return NextResponse.json(
        { error: `You already have a ${exchange} connection named "${name}"` },
        { status: 409 }
      );
    }

    // Add exchange via manager
    const result = await addExchangeAPI({
      userId: user.id,
      exchange,
      name,
      credentials: {
        apiKey,
        apiSecret,
        passphrase,
        testnet,
      },
      termsAccepted,
      disclaimerAccepted,
    });

    if (!result.success) {
      return NextResponse.json(
        { error: result.error || 'Failed to add exchange' },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      exchangeId: result.exchangeId,
      message: `${exchange.toUpperCase()} connection added successfully`,
    });
  } catch (error: any) {
    console.error('POST /api/exchanges error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to add exchange' },
      { status: 500 }
    );
  }
}
