import { NextRequest, NextResponse } from 'next/server';
// DISABLED: Missing module @/lib/azure-signalr-service
// import AzureSignalRService from '@/lib/azure-signalr-service';

/**
 * SIGNALR NEGOTIATE API
 * Get SignalR connection info for client
 */

export async function POST(_request: NextRequest) {
  // DISABLED: Missing @/lib/azure-signalr-service module
  return NextResponse.json(
    {
      success: false,
      error: 'SignalR endpoint is currently disabled. Missing backend dependencies.',
    },
    { status: 503 }
  );

  /* COMMENTED OUT UNTIL BACKEND IS READY
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
  */
}

export async function GET(request: NextRequest) {
  return POST(request);
}
