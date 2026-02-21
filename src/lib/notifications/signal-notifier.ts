/**
 * REAL-TIME SIGNAL NOTIFIER
 * Browser notifications for trading signals
 */

export interface SignalNotificationOptions {
  symbol: string;
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  strategy?: string;
}

class SignalNotifier {
  private enabled: boolean = false;
  private lastNotifications: Map<string, number> = new Map();
  private cooldownMs: number = 60000; // 1 minute cooldown per symbol

  constructor() {
    if (typeof window !== 'undefined') {
      this.enabled = this.checkPermission();
      this.loadPreferences();
    }
  }

  private checkPermission(): boolean {
    if ('Notification' in window) {
      return Notification.permission === 'granted';
    }
    return false;
  }

  private loadPreferences(): void {
    try {
      const saved = localStorage.getItem('notificationsEnabled');
      this.enabled = saved === 'true' && this.checkPermission();
    } catch (e) {
      console.error('Failed to load notification preferences:', e);
    }
  }

  async requestPermission(): Promise<boolean> {
    if (!('Notification' in window)) {
      console.warn('Browser does not support notifications');
      return false;
    }

    try {
      const permission = await Notification.requestPermission();
      this.enabled = permission === 'granted';

      if (this.enabled) {
        localStorage.setItem('notificationsEnabled', 'true');
        // Send test notification
        this.sendTestNotification();
      }

      return this.enabled;
    } catch (error) {
      console.error('Failed to request notification permission:', error);
      return false;
    }
  }

  private sendTestNotification(): void {
    new Notification('SARDAG Trading Scanner', {
      body: 'Notifications enabled! You will receive alerts for strong signals.',
      icon: '/favicon.ico',
      badge: '/favicon.ico',
    });
  }

  private shouldNotify(symbol: string): boolean {
    if (!this.enabled) return false;

    const lastTime = this.lastNotifications.get(symbol);
    if (lastTime && Date.now() - lastTime < this.cooldownMs) {
      return false; // Cooldown active
    }

    return true;
  }

  notifySignal(options: SignalNotificationOptions): void {
    if (!this.shouldNotify(options.symbol)) return;

    // Only notify for strong signals
    if (options.type === 'BUY' && options.confidence < 75) return;
    if (options.type === 'SELL' && options.confidence < 70) return;

    const title = `${options.type} Signal: ${options.symbol}`;
    const body = `Confidence: ${options.confidence}%\\nPrice: $${options.price.toFixed(6)}${options.strategy ? `\\nStrategy: ${options.strategy}` : ''}`;

    try {
      const notification = new Notification(title, {
        body,
        icon: '/favicon.ico',
        badge: '/favicon.ico',
        tag: options.symbol, // Replace previous notifications for same symbol
        requireInteraction: options.type === 'SELL', // Keep SELL signals visible
      });

      notification.onclick = () => {
        window.focus();
        window.location.href = `/market-scanner?symbol=${options.symbol}`;
        notification.close();
      };

      this.lastNotifications.set(options.symbol, Date.now());
    } catch (error) {
      console.error('Failed to show notification:', error);
    }
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled && this.checkPermission();
    localStorage.setItem('notificationsEnabled', String(this.enabled));
  }

  isEnabled(): boolean {
    return this.enabled;
  }
}

// Singleton instance
export const signalNotifier = new SignalNotifier();
