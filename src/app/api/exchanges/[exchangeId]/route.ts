/**
 * Exchange API Management - Individual Exchange
 *
 * White-hat compliance: User manages specific exchange connection
 * GET: Get exchange details
 * DELETE: Remove exchange connection
 * PATCH: Update exchange settings
 */

import { NextRequest, NextResponse } from 'next/server';
import { requirePayment } from '@/lib/auth/helpers';
import { prisma } from '@/lib/prisma';
import { deleteExchangeAPI, retestExchangeConnection } from '@/lib/exchanges/manager';

/**
 * GET /api/exchanges/[exchangeId]
 * Get specific exchange details
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ exchangeId: string }> }
) {
  try {
    const user = await requirePayment();
    const { exchangeId } = await params;

    const exchange = await prisma.exchangeAPI.findFirst({
      where: {
        id: exchangeId,
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
        testnet: true,
        createdAt: true,
        updatedAt: true,
        // NEVER return encrypted credentials
      },
    });

    if (!exchange) {
      return NextResponse.json(
        { error: 'Exchange not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      exchange,
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch exchange' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/exchanges/[exchangeId]
 * Remove exchange connection
 */
export async function DELETE(
  _req: NextRequest,
  { params }: { params: Promise<{ exchangeId: string }> }
) {
  try {
    const user = await requirePayment();
    const { exchangeId } = await params;

    // Check if exchange has active trading bots
    const activeBots = await prisma.tradingBot.count({
      where: {
        exchangeId,
        status: 'RUNNING',
      },
    });

    if (activeBots > 0) {
      return NextResponse.json(
        { error: `Cannot delete exchange with ${activeBots} active trading bot(s). Stop bots first.` },
        { status: 400 }
      );
    }

    const success = await deleteExchangeAPI(exchangeId, user.id);

    if (!success) {
      return NextResponse.json(
        { error: 'Failed to delete exchange' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Exchange connection removed successfully',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to delete exchange' },
      { status: 500 }
    );
  }
}

/**
 * PATCH /api/exchanges/[exchangeId]
 * Update exchange settings (name, active status)
 */
export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ exchangeId: string }> }
) {
  try {
    const user = await requirePayment();
    const { exchangeId } = await params;
    const body = await req.json();

    const { name, isActive } = body;

    // Build update object
    const updateData: any = {};
    if (name !== undefined) updateData.name = name;
    if (isActive !== undefined) updateData.isActive = isActive;

    if (Object.keys(updateData).length === 0) {
      return NextResponse.json(
        { error: 'No update fields provided' },
        { status: 400 }
      );
    }

    const exchange = await prisma.exchangeAPI.updateMany({
      where: {
        id: exchangeId,
        userId: user.id,
      },
      data: updateData,
    });

    if (exchange.count === 0) {
      return NextResponse.json(
        { error: 'Exchange not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      message: 'Exchange updated successfully',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to update exchange' },
      { status: 500 }
    );
  }
}
