/**
 * GLOBAL NOTIFICATION STORE
 * Real-time notification management system
 * Beyaz Şapka: Educational purposes only
 */

export type NotificationType =
  | 'SIGNAL'      // Trading signals
  | 'BOT'         // Bot status changes
  | 'NEWS'        // Important news
  | 'WHALE'       // Whale alerts
  | 'PRICE'       // Price alerts
  | 'RISK'        // Risk warnings
  | 'SYSTEM'      // System notifications
  | 'SUCCESS'     // Success messages
  | 'ERROR';      // Error messages

export type NotificationPriority = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export interface Notification {
  id: string;
  type: NotificationType;
  priority: NotificationPriority;
  title: string;
  message: string;
  data?: any;
  source?: string;
  timestamp: number;
  read: boolean;
  actionUrl?: string;
}

interface NotificationListener {
  id: string;
  callback: (notification: Notification) => void;
}

class NotificationStoreClass {
  private notifications: Notification[] = [];
  private listeners: NotificationListener[] = [];
  private maxNotifications = 100;
  private storageKey = 'ailydian_notifications';

  constructor() {
    if (typeof window !== 'undefined') {
      this.loadFromStorage();
    }
  }

  /**
   * Add a new notification
   */
  add(notification: Omit<Notification, 'id' | 'timestamp' | 'read'>): Notification {
    const newNotification: Notification = {
      ...notification,
      id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      read: false
    };

    // Add to beginning of array (newest first)
    this.notifications.unshift(newNotification);

    // Limit total notifications
    if (this.notifications.length > this.maxNotifications) {
      this.notifications = this.notifications.slice(0, this.maxNotifications);
    }

    // Save to storage
    this.saveToStorage();

    // Notify all listeners
    this.notifyListeners(newNotification);

    // Show browser notification if supported and high priority
    if (notification.priority === 'HIGH' || notification.priority === 'CRITICAL') {
      this.showBrowserNotification(newNotification);
    }

    return newNotification;
  }

  /**
   * Get all notifications
   */
  getAll(): Notification[] {
    return [...this.notifications];
  }

  /**
   * Get unread notifications
   */
  getUnread(): Notification[] {
    return this.notifications.filter(n => !n.read);
  }

  /**
   * Get unread count
   */
  getUnreadCount(): number {
    return this.notifications.filter(n => !n.read).length;
  }

  /**
   * Mark notification as read
   */
  markAsRead(id: string): void {
    const notification = this.notifications.find(n => n.id === id);
    if (notification) {
      notification.read = true;
      this.saveToStorage();
      this.notifyListeners(notification);
    }
  }

  /**
   * Mark all as read
   */
  markAllAsRead(): void {
    this.notifications.forEach(n => n.read = true);
    this.saveToStorage();
    this.listeners.forEach(listener => {
      this.notifications.forEach(n => listener.callback(n));
    });
  }

  /**
   * Delete a notification
   */
  delete(id: string): void {
    const index = this.notifications.findIndex(n => n.id === id);
    if (index !== -1) {
      this.notifications.splice(index, 1);
      this.saveToStorage();
    }
  }

  /**
   * Clear all notifications
   */
  clearAll(): void {
    this.notifications = [];
    this.saveToStorage();
  }

  /**
   * Subscribe to notifications
   */
  subscribe(callback: (notification: Notification) => void): () => void {
    const listener: NotificationListener = {
      id: `listener_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      callback
    };

    this.listeners.push(listener);

    // Return unsubscribe function
    return () => {
      const index = this.listeners.findIndex(l => l.id === listener.id);
      if (index !== -1) {
        this.listeners.splice(index, 1);
      }
    };
  }

  /**
   * Notify all listeners
   */
  private notifyListeners(notification: Notification): void {
    this.listeners.forEach(listener => {
      try {
        listener.callback(notification);
      } catch (err) {
        console.error('[NotificationStore] Listener error:', err);
      }
    });
  }

  /**
   * Save to localStorage
   */
  private saveToStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this.notifications));
    } catch (err) {
      console.error('[NotificationStore] Storage error:', err);
    }
  }

  /**
   * Load from localStorage
   */
  private loadFromStorage(): void {
    if (typeof window === 'undefined') return;

    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        this.notifications = JSON.parse(stored);

        // Remove old notifications (older than 24 hours)
        const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
        this.notifications = this.notifications.filter(n => n.timestamp > oneDayAgo);
      }
    } catch (err) {
      console.error('[NotificationStore] Load error:', err);
      this.notifications = [];
    }
  }

  /**
   * Show browser notification
   */
  private async showBrowserNotification(notification: Notification): Promise<void> {
    if (typeof window === 'undefined' || !('Notification' in window)) return;

    try {
      if (Notification.permission === 'granted') {
        new Notification(notification.title, {
          body: notification.message,
          icon: '/icons/icon-192x192.png',
          badge: '/icons/icon-96x96.png',
          tag: notification.id,
          requireInteraction: notification.priority === 'CRITICAL',
        });
      } else if (Notification.permission !== 'denied') {
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
          this.showBrowserNotification(notification);
        }
      }
    } catch (err) {
      console.error('[NotificationStore] Browser notification error:', err);
    }
  }
}

// Export singleton instance
export const NotificationStore = new NotificationStoreClass();
