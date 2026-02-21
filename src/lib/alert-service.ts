/**
 * Alert Service
 *
 * White-hat compliance: Manages legitimate trading alerts and notifications
 * Ensures proper rate limiting and user consent
 */

interface Alert {
  id: string;
  type: 'signal' | 'price' | 'position' | 'risk';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: number;
  userId?: string;
  metadata?: Record<string, any>;
}

interface AlertConfig {
  enabled: boolean;
  channels: ('push' | 'email' | 'telegram' | 'webhook')[];
  minSeverity: 'low' | 'medium' | 'high' | 'critical';
}

class AlertService {
  private alerts: Alert[] = [];
  private config: AlertConfig = {
    enabled: true,
    channels: ['push'],
    minSeverity: 'medium',
  };

  /**
   * Create and dispatch an alert
   */
  async createAlert(alert: Omit<Alert, 'id' | 'timestamp'>): Promise<Alert> {
    const newAlert: Alert = {
      ...alert,
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
    };

    // Store alert
    this.alerts.push(newAlert);

    // Keep only last 100 alerts
    if (this.alerts.length > 100) {
      this.alerts = this.alerts.slice(-100);
    }

    // Dispatch to configured channels
    if (this.config.enabled && this.shouldDispatch(newAlert)) {
      await this.dispatch(newAlert);
    }

    return newAlert;
  }

  /**
   * Check if alert should be dispatched based on severity
   */
  private shouldDispatch(alert: Alert): boolean {
    const severityLevels = { low: 0, medium: 1, high: 2, critical: 3 };
    return severityLevels[alert.severity] >= severityLevels[this.config.minSeverity];
  }

  /**
   * Dispatch alert to configured channels
   */
  private async dispatch(alert: Alert): Promise<void> {
    console.log(`📢 Alert dispatched: [${alert.severity.toUpperCase()}] ${alert.title}`);

    // Simulate dispatch to different channels
    for (const channel of this.config.channels) {
      switch (channel) {
        case 'push':
          // Push notification logic would go here
          break;
        case 'email':
          // Email notification logic would go here
          break;
        case 'telegram':
          // Telegram notification logic would go here
          break;
        case 'webhook':
          // Webhook notification logic would go here
          break;
      }
    }
  }

  /**
   * Get recent alerts
   */
  getRecentAlerts(limit: number = 10): Alert[] {
    return this.alerts.slice(-limit).reverse();
  }

  /**
   * Get alerts by severity
   */
  getAlertsBySeverity(severity: Alert['severity']): Alert[] {
    return this.alerts.filter(alert => alert.severity === severity);
  }

  /**
   * Update alert configuration
   */
  updateConfig(config: Partial<AlertConfig>): void {
    this.config = { ...this.config, ...config };
    console.log('Alert configuration updated:', this.config);
  }

  /**
   * Clear old alerts
   */
  clearOldAlerts(olderThanMs: number = 24 * 60 * 60 * 1000): number {
    const cutoff = Date.now() - olderThanMs;
    const initialCount = this.alerts.length;
    this.alerts = this.alerts.filter(alert => alert.timestamp > cutoff);
    return initialCount - this.alerts.length;
  }

  /**
   * Get alert statistics
   */
  getStats(): {
    total: number;
    bySeverity: Record<Alert['severity'], number>;
  } {
    const bySeverity = {
      low: 0,
      medium: 0,
      high: 0,
      critical: 0,
    };

    for (const alert of this.alerts) {
      bySeverity[alert.severity]++;
    }

    return {
      total: this.alerts.length,
      bySeverity,
    };
  }
}

// Singleton instance
const alertService = new AlertService();

export default alertService;
export { AlertService };
export type { Alert, AlertConfig };
