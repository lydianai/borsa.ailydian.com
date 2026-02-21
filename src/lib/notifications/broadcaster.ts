/**
 * NOTIFICATION BROADCASTER
 * Centralized notification management for SSE connections
 */

export interface NotificationEvent {
  id: string;
  type: 'signal' | 'top10' | 'ai-update' | 'quantum-update' | 'system';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  symbol?: string;
  data?: any;
  timestamp: string;
}

// In-memory notification queue (in production, use Redis or similar)
export const notificationQueue: NotificationEvent[] = [];

// Active SSE connections
export const connections: Set<ReadableStreamDefaultController> = new Set();

/**
 * Broadcast notification to all connected SSE clients
 */
export function broadcastNotification(notification: NotificationEvent) {
  notificationQueue.push(notification);

  // Keep only last 50 notifications
  if (notificationQueue.length > 50) {
    notificationQueue.shift();
  }

  // Send to all connected clients
  const message = `data: ${JSON.stringify(notification)}\n\n`;
  connections.forEach((controller) => {
    try {
      controller.enqueue(new TextEncoder().encode(message));
    } catch (error) {
      console.error('[Notifications] Error sending to client:', error);
      connections.delete(controller);
    }
  });

  console.log(`[Notifications] Broadcasted to ${connections.size} clients:`, notification.title);
}

/**
 * Get all active notifications
 */
export function getActiveNotifications(): NotificationEvent[] {
  return [...notificationQueue];
}

/**
 * Clear all notifications
 */
export function clearNotifications() {
  notificationQueue.length = 0;
}

/**
 * Get connection count
 */
export function getConnectionCount(): number {
  return connections.size;
}
