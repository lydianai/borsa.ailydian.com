/**
 * System Alert Utilities
 * Helper functions for sending system alerts to Telegram
 */

/**
 * Send critical alert to Telegram
 * Can be imported and called from anywhere in the app
 */
export async function sendCriticalAlert(title: string, message: string, details?: any) {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/telegram/system-alerts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'critical',
        title,
        message,
        source: 'System Monitor',
        details
      })
    });

    return await response.json();
  } catch (error) {
    console.error('[System Alerts] Failed to send critical alert:', error);
    return { success: false, error };
  }
}

/**
 * Send error alert to Telegram
 */
export async function sendErrorAlert(title: string, message: string, details?: any) {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/telegram/system-alerts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'error',
        title,
        message,
        source: 'System Monitor',
        details
      })
    });

    return await response.json();
  } catch (error) {
    console.error('[System Alerts] Failed to send error alert:', error);
    return { success: false, error };
  }
}

/**
 * Send warning alert to Telegram
 */
export async function sendWarningAlert(title: string, message: string, details?: any) {
  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/api/telegram/system-alerts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'warning',
        title,
        message,
        source: 'System Monitor',
        details
      })
    });

    return await response.json();
  } catch (error) {
    console.error('[System Alerts] Failed to send warning alert:', error);
    return { success: false, error };
  }
}
