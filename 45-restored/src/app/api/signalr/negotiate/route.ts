import { NextRequest, NextResponse } from 'next/server';
import AzureSignalRService from '@/lib/azure-signalr-service';

/**
 * SIGNALR NEGOTIATE API
 * Get SignalR connection info for client
 */

export async function POST(request: NextRequest) {
  try {
    const signalR = AzureSignalRService.getInstance();
    const connectionInfo = await signalR.getClientConnectionInfo();

    if (!connectionInfo) {
      return NextResponse.json(
        {
          success: false,
          error: 'SignalR not configured. Set AZURE_SIGNALR_CONN in .env',
        },
        { status: 503 }
      );
    }

    return NextResponse.json({
      success: true,
      ...connectionInfo,
    });
  } catch (error: any) {
    console.error('SignalR negotiate error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to negotiate SignalR connection',
      },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  return POST(request);
}
