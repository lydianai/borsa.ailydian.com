/**
 * ON-CHAIN WHALE NOTIFICATIONS
 * Monitors whale movements and sends advance warnings
 *
 * Features:
 * - Real-time whale activity monitoring
 * - Configurable thresholds
 * - Integration with existing notification system
 * - Non-intrusive (won't spam users)
 *
 * Integration: Uses existing signal-notifier.ts infrastructure
 */

import { getWhaleActivity, type WhaleActivity } from './index';

// ============================================================================
// TYPES
// ============================================================================

export interface WhaleNotification {
  type: 'whale-accumulation' | 'whale-distribution' | 'whale-critical';
  symbol: string;
  title: string;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  whaleActivity: WhaleActivity;
  timestamp: Date;
}

export interface WhaleNotificationConfig {
  enabled: boolean;
  minConfidence: number; // Minimum whale confidence to trigger notification (0-100)
  minNetflow: number; // Minimum USD netflow to trigger notification
  criticalRiskThreshold: number; // Risk score to trigger critical alerts
  cooldownMinutes: number; // Minutes to wait before re-notifying same symbol
}

// ============================================================================
// CONFIGURATION
// ============================================================================

const DEFAULT_CONFIG: WhaleNotificationConfig = {
  enabled: true,
  minConfidence: 60, // Only notify on 60%+ confidence whale movements
  minNetflow: 1000000, // $1M minimum netflow
  criticalRiskThreshold: 85, // Risk score 85+ triggers critical alerts
  cooldownMinutes: 30, // Don't spam - 30 min cooldown per symbol
};

// Cooldown tracker (symbol -> last notification timestamp)
const notificationCooldown = new Map<string, number>();

// ============================================================================
// NOTIFICATION DETECTION
// ============================================================================

/**
 * Check for whale activities that should trigger notifications
 */
export async function checkWhaleNotifications(
  config: Partial<WhaleNotificationConfig> = {}
): Promise<WhaleNotification[]> {
  const fullConfig = { ...DEFAULT_CONFIG, ...config };

  if (!fullConfig.enabled) {
    return [];
  }

  try {
    const whaleActivity = await getWhaleActivity();
    const notifications: WhaleNotification[] = [];
    const now = Date.now();

    whaleActivity.forEach((activity, symbol) => {
      // Check cooldown
      const lastNotification = notificationCooldown.get(symbol) || 0;
      const cooldownMs = fullConfig.cooldownMinutes * 60 * 1000;
      if (now - lastNotification < cooldownMs) {
        return; // Skip - in cooldown period
      }

      // Check if activity meets notification thresholds
      const meetsConfidence = activity.confidence >= fullConfig.minConfidence;
      const meetsNetflow = Math.abs(activity.exchangeNetflow) >= fullConfig.minNetflow;
      const isCritical = activity.riskScore >= fullConfig.criticalRiskThreshold;

      if (!meetsConfidence || !meetsNetflow) {
        return; // Not significant enough
      }

      // Create notification based on activity type
      let notification: WhaleNotification | null = null;

      if (isCritical) {
        // Critical alert
        notification = {
          type: 'whale-critical',
          symbol,
          title: `🚨 CRITICAL: ${symbol} Whale Alert`,
          message: `Extreme whale activity detected! ${activity.summary}. Risk Score: ${activity.riskScore}/100`,
          severity: 'critical',
          whaleActivity: activity,
          timestamp: new Date(),
        };
      } else if (activity.activity === 'accumulation') {
        // Bullish whale accumulation
        notification = {
          type: 'whale-accumulation',
          symbol,
          title: `🐋 ${symbol} - Whales Accumulating`,
          message: `${activity.summary}. Confidence: ${activity.confidence}%. This could signal upcoming price increase.`,
          severity: 'info',
          whaleActivity: activity,
          timestamp: new Date(),
        };
      } else if (activity.activity === 'distribution') {
        // Bearish whale distribution
        notification = {
          type: 'whale-distribution',
          symbol,
          title: `⚠️ ${symbol} - Whales Distributing`,
          message: `${activity.summary}. Confidence: ${activity.confidence}%. This could signal upcoming price decrease.`,
          severity: 'warning',
          whaleActivity: activity,
          timestamp: new Date(),
        };
      }

      if (notification) {
        notifications.push(notification);
        notificationCooldown.set(symbol, now); // Update cooldown
      }
    });

    if (notifications.length > 0) {
      console.log(`[WhaleNotifications] Generated ${notifications.length} whale notifications`);
    }

    return notifications;
  } catch (error) {
    console.error('[WhaleNotifications] Error checking whale notifications:', error);
    return [];
  }
}

/**
 * Get whale notification summary (for UI display)
 */
export async function getWhaleNotificationSummary(
  config: Partial<WhaleNotificationConfig> = {}
): Promise<{
  total: number;
  byType: {
    accumulation: number;
    distribution: number;
    critical: number;
  };
  notifications: WhaleNotification[];
}> {
  const notifications = await checkWhaleNotifications(config);

  const summary = {
    total: notifications.length,
    byType: {
      accumulation: notifications.filter((n) => n.type === 'whale-accumulation').length,
      distribution: notifications.filter((n) => n.type === 'whale-distribution').length,
      critical: notifications.filter((n) => n.type === 'whale-critical').length,
    },
    notifications,
  };

  return summary;
}

/**
 * Monitor specific symbols for whale activity
 */
export async function monitorSymbolsForWhaleActivity(
  symbols: string[],
  config: Partial<WhaleNotificationConfig> = {}
): Promise<Map<string, WhaleNotification | null>> {
  const fullConfig = { ...DEFAULT_CONFIG, ...config };
  const results = new Map<string, WhaleNotification | null>();

  try {
    const whaleActivity = await getWhaleActivity();

    for (const symbol of symbols) {
      const activity = whaleActivity.get(symbol.toUpperCase());

      if (!activity) {
        results.set(symbol, null);
        continue;
      }

      // Check thresholds
      const meetsConfidence = activity.confidence >= fullConfig.minConfidence;
      const meetsNetflow = Math.abs(activity.exchangeNetflow) >= fullConfig.minNetflow;
      const isCritical = activity.riskScore >= fullConfig.criticalRiskThreshold;

      if (!meetsConfidence || !meetsNetflow) {
        results.set(symbol, null);
        continue;
      }

      // Create notification
      let notification: WhaleNotification | null = null;

      if (isCritical) {
        notification = {
          type: 'whale-critical',
          symbol,
          title: `🚨 CRITICAL: ${symbol} Whale Alert`,
          message: `Extreme whale activity! ${activity.summary}`,
          severity: 'critical',
          whaleActivity: activity,
          timestamp: new Date(),
        };
      } else if (activity.activity === 'accumulation') {
        notification = {
          type: 'whale-accumulation',
          symbol,
          title: `🐋 ${symbol} - Whales Buying`,
          message: activity.summary,
          severity: 'info',
          whaleActivity: activity,
          timestamp: new Date(),
        };
      } else if (activity.activity === 'distribution') {
        notification = {
          type: 'whale-distribution',
          symbol,
          title: `⚠️ ${symbol} - Whales Selling`,
          message: activity.summary,
          severity: 'warning',
          whaleActivity: activity,
          timestamp: new Date(),
        };
      }

      results.set(symbol, notification);
    }
  } catch (error) {
    console.error('[WhaleNotifications] Error monitoring symbols:', error);
  }

  return results;
}

/**
 * Clear notification cooldown for a symbol (for testing)
 */
export function clearNotificationCooldown(symbol?: string): void {
  if (symbol) {
    notificationCooldown.delete(symbol.toUpperCase());
    console.log(`[WhaleNotifications] Cooldown cleared for ${symbol}`);
  } else {
    notificationCooldown.clear();
    console.log('[WhaleNotifications] All cooldowns cleared');
  }
}

/**
 * Get notification cooldown status
 */
export function getNotificationCooldownStatus(): Map<string, { lastNotification: Date; cooldownRemaining: number }> {
  const status = new Map<string, { lastNotification: Date; cooldownRemaining: number }>();
  const now = Date.now();
  const cooldownMs = DEFAULT_CONFIG.cooldownMinutes * 60 * 1000;

  notificationCooldown.forEach((timestamp, symbol) => {
    const remaining = Math.max(0, cooldownMs - (now - timestamp));
    status.set(symbol, {
      lastNotification: new Date(timestamp),
      cooldownRemaining: Math.round(remaining / 1000 / 60), // Minutes
    });
  });

  return status;
}

// ============================================================================
// INTEGRATION WITH EXISTING NOTIFICATION SYSTEM
// ============================================================================

/**
 * Convert whale notification to browser notification format
 * Compatible with existing Web Push Notifications
 */
export function toWebPushNotification(whaleNotification: WhaleNotification): {
  title: string;
  body: string;
  icon?: string;
  badge?: string;
  tag: string;
  data: any;
} {
  let icon = '/icon-192x192.png';
  let badge = '/icon-96x96.png';

  if (whaleNotification.severity === 'critical') {
    icon = '/icon-close-96x96.png'; // Red icon for critical
  } else if (whaleNotification.type === 'whale-accumulation') {
    icon = '/icon-chart-96x96.png'; // Chart icon for accumulation
  }

  return {
    title: whaleNotification.title,
    body: whaleNotification.message,
    icon,
    badge,
    tag: `whale-${whaleNotification.symbol}-${whaleNotification.type}`,
    data: {
      type: whaleNotification.type,
      symbol: whaleNotification.symbol,
      severity: whaleNotification.severity,
      timestamp: whaleNotification.timestamp,
      whaleActivity: {
        activity: whaleNotification.whaleActivity.activity,
        confidence: whaleNotification.whaleActivity.confidence,
        riskScore: whaleNotification.whaleActivity.riskScore,
        exchangeNetflow: whaleNotification.whaleActivity.exchangeNetflow,
      },
    },
  };
}

/**
 * Format whale notification for in-app display
 */
export function formatForInAppNotification(whaleNotification: WhaleNotification): {
  id: string;
  type: string;
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
  icon: string;
  color: string;
  action?: {
    label: string;
    url: string;
  };
} {
  let color = '#3b82f6'; // Blue (info)
  let icon = '🐋';

  if (whaleNotification.severity === 'critical') {
    color = '#ef4444'; // Red
    icon = '🚨';
  } else if (whaleNotification.severity === 'warning') {
    color = '#f59e0b'; // Orange
    icon = '⚠️';
  }

  return {
    id: `whale-${whaleNotification.symbol}-${Date.now()}`,
    type: whaleNotification.type,
    title: whaleNotification.title,
    message: whaleNotification.message,
    timestamp: whaleNotification.timestamp,
    read: false,
    icon,
    color,
    action: {
      label: `View ${whaleNotification.symbol}`,
      url: `/?symbol=${whaleNotification.symbol}`,
    },
  };
}
