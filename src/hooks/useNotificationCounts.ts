/**
 * 🔔 USE NOTIFICATION COUNTS HOOK
 * Reusable hook to read notification counts from localStorage
 * Used by all pages to display notification badges in SharedSidebar
 */

import { useState, useEffect } from 'react';

export interface NotificationCounts {
  market: number;
  trading: number;
  ai: number;
  quantum: number;
  conservative: number;
  omnipotent: number;
  correlation: number;
  btceth: number;
  traditional: number;
}

export function useNotificationCounts(): NotificationCounts {
  const [counts, setCounts] = useState<NotificationCounts>({
    market: 0,
    trading: 0,
    ai: 0,
    quantum: 0,
    conservative: 0,
    omnipotent: 0,
    correlation: 0,
    btceth: 0,
    traditional: 0,
  });

  useEffect(() => {
    const loadAllNotifications = () => {
      if (typeof window !== 'undefined') {
        const market = localStorage.getItem('market_notification_count');
        const trading = localStorage.getItem('trading_notification_count');
        const ai = localStorage.getItem('ai_notification_count');
        const quantum = localStorage.getItem('quantum_notification_count');
        const conservative = localStorage.getItem('conservative_notification_count');
        const omnipotent = localStorage.getItem('omnipotent_notification_count');
        const correlation = localStorage.getItem('correlation_notification_count');
        const btceth = localStorage.getItem('btceth_notification_count');
        const traditional = localStorage.getItem('traditional_notification_count');

        setCounts({
          market: market ? parseInt(market) : 0,
          trading: trading ? parseInt(trading) : 0,
          ai: ai ? parseInt(ai) : 0,
          quantum: quantum ? parseInt(quantum) : 0,
          conservative: conservative ? parseInt(conservative) : 0,
          omnipotent: omnipotent ? parseInt(omnipotent) : 0,
          correlation: correlation ? parseInt(correlation) : 0,
          btceth: btceth ? parseInt(btceth) : 0,
          traditional: traditional ? parseInt(traditional) : 0,
        });
      }
    };

    // Initial load
    loadAllNotifications();

    // Listen for storage changes (when other tabs/windows update counts)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key?.includes('notification_count') && e.newValue) {
        loadAllNotifications();
      }
    };

    window.addEventListener('storage', handleStorageChange);

    // Check periodically every 2 seconds
    const interval = setInterval(loadAllNotifications, 2000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      clearInterval(interval);
    };
  }, []);

  return counts;
}
