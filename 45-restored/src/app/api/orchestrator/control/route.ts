import { NextRequest, NextResponse } from 'next/server';
import { getBotIntegrationManager } from '@/services/orchestrator/BotIntegrationManager';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    if (action !== 'start' && action !== 'stop') {
      return NextResponse.json(
        { success: false, error: 'Action must be "start" or "stop"' },
        { status: 400 }
      );
    }

    const manager = getBotIntegrationManager();

    if (action === 'start') {
      await manager.initialize();
      return NextResponse.json({
        success: true,
        message: 'Orchestrator started successfully',
        timestamp: Date.now()
      });
    } else {
      await manager.shutdown();
      return NextResponse.json({
        success: true,
        message: 'Orchestrator stopped successfully',
        timestamp: Date.now()
      });
    }
  } catch (error) {
    console.error('[API] Control orchestrator error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const manager = getBotIntegrationManager();
    const status = manager.getStatus();

    return NextResponse.json({
      success: true,
      status,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('[API] Get control status error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
