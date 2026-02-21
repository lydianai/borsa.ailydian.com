/**
 * 📡 REAL-TIME OTONOM SİSTEM HOOK
 *
 * SSE stream'ini dinler ve otomatik olarak state'i günceller.
 * Frontend componentlerde kullanmak için.
 *
 * Kullanım:
 * ```tsx
 * const { health, queueStatus, isConnected, error } = useAutonomousSystem();
 * ```
 */

import { useState, useEffect, useCallback, useRef } from 'react';

export interface AutonomousHealth {
  timestamp: string;
  redis: {
    status: 'healthy' | 'unhealthy';
    latency?: number;
    error?: string;
  };
  queue: {
    status: 'healthy' | 'unhealthy';
    active?: number;
    waiting?: number;
    completed?: number;
    failed?: number;
    error?: string;
  };
  cron: {
    status: 'healthy' | 'unhealthy';
    totalJobs?: number;
    error?: string;
  };
}

export interface QueueStatus {
  timestamp: string;
  status: 'healthy' | 'unhealthy';
  active: number;
  waiting: number;
  completed: number;
  failed: number;
}

export interface HeartbeatData {
  timestamp: string;
  uptime: number;
}

export function useAutonomousSystem() {
  const [isConnected, setIsConnected] = useState(false);
  const [health, setHealth] = useState<AutonomousHealth | null>(null);
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null);
  const [heartbeat, setHeartbeat] = useState<HeartbeatData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    // Cleanup existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    try {
      console.log('🔌 Connecting to autonomous system stream...');

      const eventSource = new EventSource('/api/stream/autonomous');
      eventSourceRef.current = eventSource;

      // Connected event
      eventSource.addEventListener('connected', (e) => {
        const data = JSON.parse(e.data);
        console.log('✅ Connected to autonomous system:', data.message);
        setIsConnected(true);
        setError(null);
        setReconnectAttempts(0);
      });

      // Health updates
      eventSource.addEventListener('health', (e) => {
        const data: AutonomousHealth = JSON.parse(e.data);
        setHealth(data);
      });

      // Queue status updates
      eventSource.addEventListener('queue_status', (e) => {
        const data: QueueStatus = JSON.parse(e.data);
        setQueueStatus(data);
      });

      // Heartbeat
      eventSource.addEventListener('heartbeat', (e) => {
        const data: HeartbeatData = JSON.parse(e.data);
        setHeartbeat(data);
      });

      // Error event from stream
      eventSource.addEventListener('error_event', (e) => {
        const data = JSON.parse(e.data);
        console.error('❌ Stream error:', data.message);
        setError(data.message);
      });

      // Connection error
      eventSource.onerror = (err) => {
        console.error('❌ EventSource error:', err);
        setIsConnected(false);
        setError('Connection lost');

        // Auto-reconnect with exponential backoff
        const backoffTime = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000); // Max 30 seconds
        console.log(`🔄 Reconnecting in ${backoffTime / 1000}s...`);

        reconnectTimeoutRef.current = setTimeout(() => {
          setReconnectAttempts((prev) => prev + 1);
          connect();
        }, backoffTime);
      };
    } catch (err: any) {
      console.error('❌ Failed to create EventSource:', err);
      setError(err.message);
      setIsConnected(false);
    }
  }, [reconnectAttempts]);

  const disconnect = useCallback(() => {
    console.log('🔌 Disconnecting from autonomous system stream...');

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setIsConnected(false);
  }, []);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Manual reconnect function
  const reconnect = useCallback(() => {
    console.log('🔄 Manual reconnect triggered');
    setReconnectAttempts(0);
    disconnect();
    setTimeout(connect, 1000);
  }, [connect, disconnect]);

  return {
    isConnected,
    health,
    queueStatus,
    heartbeat,
    error,
    reconnectAttempts,
    reconnect,
    disconnect,
  };
}
