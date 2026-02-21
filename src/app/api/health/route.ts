import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    status: "ok",
    message: "Backend API is running",
    timestamp: new Date().toISOString(),
    version: "2.0.0-backend-only",
  });
}
