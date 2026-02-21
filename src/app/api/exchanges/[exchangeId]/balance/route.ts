/**
 * Exchange Balance Check
 *
 * White-hat compliance: Read-only balance retrieval
 * GET: Fetch current balance from exchange
 */

import { NextRequest, NextResponse } from 'next/server';
import { requirePayment } from '@/lib/auth/helpers';
import { getExchangeCredentials } from '@/lib/exchanges/manager';
import { prisma } from '@/lib/prisma';
import * as OKX from '@/lib/exchanges/okx';
import * as Bybit from '@/lib/exchanges/bybit';
import * as Coinbase from '@/lib/exchanges/coinbase';
import * as Kraken from '@/lib/exchanges/kraken';
import * as BTCTurk from '@/lib/exchanges/btcturk';

/**
 * GET /api/exchanges/[exchangeId]/balance
 * Get current balance from exchange
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ exchangeId: string }> }
) {
  try {
    const user = await requirePayment();
    const { exchangeId } = await params;

    // Get exchange record
    const exchange = await prisma.exchangeAPI.findFirst({
      where: {
        id: exchangeId,
        userId: user.id,
        isActive: true,
      },
    });

    if (!exchange) {
      return NextResponse.json(
        { error: 'Exchange not found' },
        { status: 404 }
      );
    }

    // Get decrypted credentials
    const credentials = await getExchangeCredentials(exchangeId, user.id);
    if (!credentials) {
      return NextResponse.json(
        { error: 'Failed to retrieve credentials' },
        { status: 500 }
      );
    }

    let balances: any[] = [];

    try {
      switch (exchange.exchange) {
        case 'okx':
          balances = await OKX.getOKXBalance({
            ...credentials,
            passphrase: credentials.passphrase!,
          } as OKX.OKXCredentials);
          break;

        case 'bybit':
          balances = await Bybit.getBybitBalance(
            credentials as Bybit.BybitCredentials
          );
          break;

        case 'coinbase':
          balances = await Coinbase.getCoinbaseBalance(
            credentials as Coinbase.CoinbaseCredentials
          );
          break;

        case 'kraken':
          balances = await Kraken.getKrakenBalance({
            apiKey: credentials.apiKey,
            privateKey: credentials.apiSecret,
          });
          break;

        case 'btcturk':
          balances = await BTCTurk.getBTCTurkBalance(
            credentials as BTCTurk.BTCTurkCredentials
          );
          break;

        default:
          return NextResponse.json(
            { error: 'Unsupported exchange' },
            { status: 400 }
          );
      }

      // Update last balance check
      await prisma.exchangeAPI.update({
        where: { id: exchangeId },
        data: {
          lastBalanceCheck: new Date(),
          isConnected: true,
          connectionError: null,
        },
      });

      return NextResponse.json({
        success: true,
        exchange: exchange.exchange,
        balances,
        timestamp: new Date().toISOString(),
      });
    } catch (balanceError: any) {
      // Update connection error
      await prisma.exchangeAPI.update({
        where: { id: exchangeId },
        data: {
          isConnected: false,
          connectionError: balanceError.message,
        },
      });

      return NextResponse.json(
        { error: `Failed to fetch balance: ${balanceError.message}` },
        { status: 500 }
      );
    }
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch balance' },
      { status: 500 }
    );
  }
}
