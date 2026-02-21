import { NextRequest, NextResponse } from 'next/server';
import { nanoid } from 'nanoid';

// ============================================
// ACTIVE USERS TRACKING
// Simple in-memory tracking for active users
// ============================================

// Store active users with their last seen timestamp
const activeUsers = new Map<string, number>();

// Cleanup interval (remove users inactive for 5 minutes)
const INACTIVE_THRESHOLD = 5 * 60 * 1000; // 5 minutes
const CLEANUP_INTERVAL = 60 * 1000; // 1 minute

// Cleanup old users periodically
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    const now = Date.now();
    for (const [userId, lastSeen] of activeUsers.entries()) {
      if (now - lastSeen > INACTIVE_THRESHOLD) {
        activeUsers.delete(userId);
      }
    }
  }, CLEANUP_INTERVAL);
}

/**
 * GET /api/active-users
 * Returns the current count of active users
 */
export async function GET(_req: NextRequest) {
  try {
    // Clean up inactive users before counting
    const now = Date.now();
    for (const [userId, lastSeen] of activeUsers.entries()) {
      if (now - lastSeen > INACTIVE_THRESHOLD) {
        activeUsers.delete(userId);
      }
    }

    const count = activeUsers.size;

    return NextResponse.json({
      success: true,
      count,
      timestamp: now,
    });
  } catch (error) {
    console.error('[ActiveUsers] Error getting count:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to get active users count' },
      { status: 500 }
    );
  }
}

/**
 * POST /api/active-users
 * Registers or updates a user's activity
 * Body: { userId?: string } - Optional userId, will be generated if not provided
 */
export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({}));

    // Get or generate user ID
    let userId = body.userId;
    if (!userId) {
      // Generate a unique session ID
      userId = nanoid();
    }

    // Update user's last seen timestamp
    activeUsers.set(userId, Date.now());

    return NextResponse.json({
      success: true,
      userId,
      count: activeUsers.size,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[ActiveUsers] Error updating activity:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to update user activity' },
      { status: 500 }
    );
  }
}

/**
 * DELETE /api/active-users
 * Removes a user from active tracking
 * Body: { userId: string }
 */
export async function DELETE(req: NextRequest) {
  try {
    const body = await req.json();
    const { userId } = body;

    if (!userId) {
      return NextResponse.json(
        { success: false, error: 'userId is required' },
        { status: 400 }
      );
    }

    activeUsers.delete(userId);

    return NextResponse.json({
      success: true,
      count: activeUsers.size,
      timestamp: Date.now(),
    });
  } catch (error) {
    console.error('[ActiveUsers] Error removing user:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to remove user' },
      { status: 500 }
    );
  }
}
