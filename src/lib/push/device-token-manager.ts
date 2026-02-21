/**
 * 📱 DEVICE TOKEN MANAGER
 * Manages FCM device tokens for push notifications
 *
 * Features:
 * - In-memory token storage (can be upgraded to database)
 * - Token validation
 * - User-device mapping
 * - Token expiration handling
 * - White-hat compliance: All operations logged
 */

// ============================================================================
// TYPES
// ============================================================================

export interface DeviceToken {
  token: string;
  userId: string;
  platform: 'ios' | 'android' | 'web';
  registeredAt: Date;
  lastUsedAt: Date;
  metadata?: {
    deviceModel?: string;
    osVersion?: string;
    appVersion?: string;
  };
}

export interface TokenStore {
  [token: string]: DeviceToken;
}

// ============================================================================
// DEVICE TOKEN MANAGER
// ============================================================================

export class DeviceTokenManager {
  private tokens: TokenStore = new Map() as any; // In-memory store
  private userTokenIndex: Map<string, Set<string>> = new Map(); // userId -> token[]

  /**
   * Register a new device token
   */
  registerToken(
    token: string,
    userId: string,
    platform: 'ios' | 'android' | 'web',
    metadata?: DeviceToken['metadata']
  ): void {
    // Validate token
    if (!token || token.length < 20) {
      throw new Error('Invalid FCM token');
    }

    if (!userId) {
      throw new Error('User ID is required');
    }

    // Check if token already exists
    const existingToken = this.tokens[token];

    if (existingToken) {
      // Update last used time
      existingToken.lastUsedAt = new Date();
      existingToken.metadata = metadata;
      console.log(`[DeviceTokenManager] Token refreshed for user ${userId}`);
      return;
    }

    // Create new token entry
    const deviceToken: DeviceToken = {
      token,
      userId,
      platform,
      registeredAt: new Date(),
      lastUsedAt: new Date(),
      metadata,
    };

    this.tokens[token] = deviceToken;

    // Update user-token index
    if (!this.userTokenIndex.has(userId)) {
      this.userTokenIndex.set(userId, new Set());
    }
    this.userTokenIndex.get(userId)!.add(token);

    console.log(
      `[DeviceTokenManager] ✅ Token registered for user ${userId} (${platform})`
    );
  }

  /**
   * Unregister a device token
   */
  unregisterToken(token: string): boolean {
    const deviceToken = this.tokens[token];

    if (!deviceToken) {
      return false;
    }

    // Remove from user-token index
    const userId = deviceToken.userId;
    if (this.userTokenIndex.has(userId)) {
      this.userTokenIndex.get(userId)!.delete(token);

      // Clean up empty sets
      if (this.userTokenIndex.get(userId)!.size === 0) {
        this.userTokenIndex.delete(userId);
      }
    }

    // Remove token
    delete this.tokens[token];

    console.log(`[DeviceTokenManager] Token unregistered for user ${userId}`);
    return true;
  }

  /**
   * Get all tokens for a user
   */
  getUserTokens(userId: string): string[] {
    const tokenSet = this.userTokenIndex.get(userId);

    if (!tokenSet || tokenSet.size === 0) {
      return [];
    }

    return Array.from(tokenSet);
  }

  /**
   * Get device token info
   */
  getTokenInfo(token: string): DeviceToken | null {
    return this.tokens[token] || null;
  }

  /**
   * Get all tokens (for broadcast)
   */
  getAllTokens(): string[] {
    return Object.keys(this.tokens);
  }

  /**
   * Clean up expired tokens
   * Remove tokens not used in the last 90 days
   */
  cleanupExpiredTokens(maxAgeDays: number = 90): number {
    const now = Date.now();
    const maxAgeMs = maxAgeDays * 24 * 60 * 60 * 1000;
    let removedCount = 0;

    for (const [token, deviceToken] of Object.entries(this.tokens)) {
      const age = now - deviceToken.lastUsedAt.getTime();

      if (age > maxAgeMs) {
        this.unregisterToken(token);
        removedCount++;
      }
    }

    if (removedCount > 0) {
      console.log(
        `[DeviceTokenManager] Cleaned up ${removedCount} expired tokens`
      );
    }

    return removedCount;
  }

  /**
   * Mark token as invalid (e.g., after failed send)
   */
  markTokenInvalid(token: string): void {
    this.unregisterToken(token);
    console.log(`[DeviceTokenManager] Token marked as invalid: ${token.substring(0, 20)}...`);
  }

  /**
   * Update last used time for a token
   */
  updateLastUsed(token: string): void {
    const deviceToken = this.tokens[token];

    if (deviceToken) {
      deviceToken.lastUsedAt = new Date();
    }
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalTokens: number;
    totalUsers: number;
    platformBreakdown: { ios: number; android: number; web: number };
  } {
    const stats = {
      totalTokens: Object.keys(this.tokens).length,
      totalUsers: this.userTokenIndex.size,
      platformBreakdown: { ios: 0, android: 0, web: 0 },
    };

    for (const token of Object.values(this.tokens)) {
      stats.platformBreakdown[token.platform]++;
    }

    return stats;
  }

  /**
   * Clear all tokens (for testing)
   */
  clearAll(): void {
    this.tokens = {};
    this.userTokenIndex.clear();
    console.log('[DeviceTokenManager] All tokens cleared');
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

const deviceTokenManager = new DeviceTokenManager();
export default deviceTokenManager;
