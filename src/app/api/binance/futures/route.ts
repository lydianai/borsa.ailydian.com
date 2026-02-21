import { NextResponse } from "next/server";
import { fetchBinanceFuturesData } from "@/lib/binance-data-fetcher";

export async function GET() {
  const result = await fetchBinanceFuturesData();

  if (!result.success) {
    return NextResponse.json(
      {
        success: false,
        error: result.error || "Failed to fetch Binance futures data",
      },
      { status: 500 }
    );
  }

  return NextResponse.json(result);
}
