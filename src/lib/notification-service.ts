/**
 * FRONTEND NOTIFICATION SERVICE
 * Connects to SSE endpoint and displays browser notifications
 */

interface NotificationEvent {
  id: string;
  type: 'signal' | 'top10' | 'ai-update' | 'quantum-update' | 'system';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  symbol?: string;
  data?: any;
  timestamp: string;
}

type NotificationCallback = (notification: NotificationEvent) => void;

class NotificationService {
  private eventSource: EventSource | null = null;
  private callbacks: Set<NotificationCallback> = new Set();
  private isConnected = false;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private notificationsEnabled = false;
  private _audio: HTMLAudioElement | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      // Check if notifications are enabled
      const savedPref = localStorage.getItem('notificationsEnabled');
      this.notificationsEnabled = savedPref === 'true';

      // Create notification sound using Web Audio API (works better than mp3)
      this.initializeAudioContext();
    }
  }

  private initializeAudioContext() {
    // Will be initialized on first user interaction (required by browsers)
  }

  connect() {
    if (typeof window === 'undefined' || this.eventSource) return;

    console.log('[NotificationService] Connecting to SSE endpoint...');

    this.eventSource = new EventSource('/api/notifications');

    this.eventSource.onopen = () => {
      console.log('[NotificationService] Connected to notification stream');
      this.isConnected = true;
      if (this.reconnectTimeout) {
        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = null;
      }
    };

    this.eventSource.onmessage = (event) => {
      try {
        const notification: NotificationEvent = JSON.parse(event.data);
        console.log('[NotificationService] Received:', notification);

        // Show browser notification if enabled
        if (
          this.notificationsEnabled &&
          this.shouldShowNotification(notification) &&
          'Notification' in window &&
          Notification.permission === 'granted'
        ) {
          this.showBrowserNotification(notification);
        }

        // Trigger callbacks
        this.callbacks.forEach((callback) => callback(notification));
      } catch (error) {
        console.error('[NotificationService] Parse error:', error);
      }
    };

    this.eventSource.onerror = (_error) => {
      console.warn('[NotificationService] SSE Connection error (normal during dev reload)');
      this.isConnected = false;
      this.eventSource?.close();
      this.eventSource = null;

      // Reconnect after 5 seconds (with exponential backoff)
      if (!this.reconnectTimeout) {
        const delay = 5000;
        console.log(`[NotificationService] Will reconnect in ${delay/1000}s...`);
        this.reconnectTimeout = setTimeout(() => {
          console.log('[NotificationService] Attempting to reconnect...');
          this.connect();
        }, delay);
      }
    };
  }

  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.isConnected = false;
    }
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  subscribe(callback: NotificationCallback) {
    this.callbacks.add(callback);
    return () => this.callbacks.delete(callback);
  }

  setEnabled(enabled: boolean) {
    this.notificationsEnabled = enabled;
    if (typeof window !== 'undefined') {
      localStorage.setItem('notificationsEnabled', String(enabled));
    }
  }

  private shouldShowNotification(notification: NotificationEvent): boolean {
    // Don't show system messages or low priority
    if (notification.type === 'system' || notification.priority === 'low') {
      return false;
    }

    // Check user preferences from localStorage
    if (typeof window !== 'undefined') {
      const settings = {
        strongBuy: localStorage.getItem('notify_strong_buy') !== 'false',
        sell: localStorage.getItem('notify_sell') !== 'false',
        aiUpdates: localStorage.getItem('notify_ai_updates') === 'true',
        quantum: localStorage.getItem('notify_quantum') === 'true',
      };

      if (notification.type === 'signal') {
        const data = notification.data;
        if (data?.type === 'BUY' && data?.confidence >= 80) {
          return settings.strongBuy;
        }
        if (data?.type === 'SELL') {
          return settings.sell;
        }
      }

      if (notification.type === 'ai-update') {
        return settings.aiUpdates;
      }

      if (notification.type === 'quantum-update') {
        return settings.quantum;
      }
    }

    return true;
  }

  private showBrowserNotification(notification: NotificationEvent) {
    // Play notification sound
    this.playNotificationSound();

    // Vibrate on mobile (if supported) - Enhanced pattern
    if ('vibrate' in navigator) {
      // More noticeable pattern: Short-Long-Short
      navigator.vibrate([150, 80, 250, 80, 150]);
      console.log('[NotificationService] 📳 Vibration triggered');
    }

    const options: NotificationOptions = {
      body: notification.message,
      icon: '/favicon.ico',
      badge: '/favicon.ico',
      tag: notification.symbol || notification.type,
      requireInteraction: notification.priority === 'critical',
      silent: false, // IMPORTANT: Ensure notification has sound
      data: {
        url: notification.symbol ? `/market-scanner?symbol=${notification.symbol}` : '/',
      },
    };

    const n = new Notification(notification.title, options);

    n.onclick = (event) => {
      event.preventDefault();
      const url = (n.data as any)?.url || '/';
      window.focus();
      window.location.href = url;
      n.close();
    };

    // Auto-close after 5 seconds for non-critical notifications
    if (notification.priority !== 'critical') {
      setTimeout(() => n.close(), 5000);
    }
  }

  private playNotificationSound() {
    try {
      // Create AudioContext on demand (browser security requirement)
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      const audioContext = new AudioContext();

      // Create 3-tone notification sound (MORE NOTICEABLE & PLEASANT)
      const oscillator1 = audioContext.createOscillator();
      const oscillator2 = audioContext.createOscillator();
      const oscillator3 = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator1.connect(gainNode);
      oscillator2.connect(gainNode);
      oscillator3.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // First tone: 880 Hz (A5) - Bright and attention-grabbing
      oscillator1.frequency.value = 880;
      oscillator1.type = 'sine';

      // Second tone: 1108 Hz (C#6) - Creates pleasant harmony
      oscillator2.frequency.value = 1108;
      oscillator2.type = 'sine';

      // Third tone: 1320 Hz (E6) - Completes the chord (A major)
      oscillator3.frequency.value = 1320;
      oscillator3.type = 'sine';

      // Enhanced envelope for more noticeable sound
      gainNode.gain.setValueAtTime(0, audioContext.currentTime);
      gainNode.gain.linearRampToValueAtTime(0.6, audioContext.currentTime + 0.03); // LOUDER (0.4 → 0.6)
      gainNode.gain.linearRampToValueAtTime(0.5, audioContext.currentTime + 0.15); // Sustain
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

      // Play all three tones with staggered start for melodic effect
      oscillator1.start(audioContext.currentTime);
      oscillator2.start(audioContext.currentTime + 0.08); // Slight delay
      oscillator3.start(audioContext.currentTime + 0.16); // More delay for arpeggio effect

      oscillator1.stop(audioContext.currentTime + 0.5);
      oscillator2.stop(audioContext.currentTime + 0.58);
      oscillator3.stop(audioContext.currentTime + 0.66);

      // Cleanup
      setTimeout(() => audioContext.close(), 750);

      console.log('[NotificationService] 🔔 Enhanced 3-tone notification sound played');
    } catch (err) {
      console.log('[NotificationService] Audio play failed:', err);
    }
  }

  // Public method to play sound manually
  playSound() {
    this.playNotificationSound();
  }

  getConnectionStatus() {
    return this.isConnected;
  }
}

// Singleton instance
export const notificationService = new NotificationService();
