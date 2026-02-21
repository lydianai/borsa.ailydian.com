'use client';

import { useEffect, useState } from 'react';
import { Icons } from '@/components/Icons';

interface ActiveUsersResponse {
  success: boolean;
  count: number;
  userId?: string;
  timestamp: number;
}

export function ActiveUsersIndicator() {
  const [activeCount, setActiveCount] = useState<number>(0);
  const [userId, setUserId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Register user and get count
  const registerActivity = async () => {
    try {
      const response = await fetch('/api/active-users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId: userId || undefined }),
      });

      if (response.ok) {
        const data: ActiveUsersResponse = await response.json();
        if (data.success) {
          setActiveCount(data.count);
          if (data.userId && !userId) {
            setUserId(data.userId);
            // Store in localStorage for persistence
            localStorage.setItem('active-user-id', data.userId);
          }
        }
      }
    } catch (error) {
      console.error('[ActiveUsers] Failed to register activity:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Get current count
  const fetchActiveCount = async () => {
    try {
      const response = await fetch('/api/active-users');
      if (response.ok) {
        const data: ActiveUsersResponse = await response.json();
        if (data.success) {
          setActiveCount(data.count);
        }
      }
    } catch (error) {
      console.error('[ActiveUsers] Failed to fetch count:', error);
    }
  };

  // Initialize
  useEffect(() => {
    // Get userId from localStorage if exists
    const storedUserId = localStorage.getItem('active-user-id');
    if (storedUserId) {
      setUserId(storedUserId);
    }

    // Register initial activity
    registerActivity();

    // Update activity every 2 minutes
    const activityInterval = setInterval(registerActivity, 2 * 60 * 1000);

    // Fetch count every 10 seconds for real-time updates
    const countInterval = setInterval(fetchActiveCount, 10 * 1000);

    // Cleanup on unmount
    return () => {
      clearInterval(activityInterval);
      clearInterval(countInterval);

      // Optionally remove user on page close (if you want immediate cleanup)
      // Note: This might not always fire reliably
      if (userId) {
        navigator.sendBeacon(
          '/api/active-users',
          JSON.stringify({ userId, _method: 'DELETE' })
        );
      }
    };
  }, [userId]);

  // Handle visibility change (tab switching)
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // User came back to tab - register activity
        registerActivity();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [userId]);

  if (isLoading) {
    return null; // or a skeleton loader
  }

  return (
    <div
      className="active-users-indicator"
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '6px 12px',
        background: 'rgba(139, 90, 60, 0.1)',
        borderRadius: '20px',
        border: '1px solid rgba(139, 90, 60, 0.3)',
        fontSize: '13px',
        fontWeight: '500',
        color: '#8B5A3C',
        transition: 'all 0.3s ease',
      }}
      title={`${activeCount} aktif kullanıcı çevrimiçi`}
    >
      <Icons.Users
        style={{
          width: '16px',
          height: '16px',
          color: '#8B5A3C',
        }}
      />
      <span
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}
      >
        <span
          style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: '#10b981',
            boxShadow: '0 0 8px rgba(16, 185, 129, 0.6)',
            animation: 'pulse 2s ease-in-out infinite',
          }}
        />
        {activeCount.toLocaleString('tr-TR')}
      </span>

      <style jsx>{`
        @keyframes pulse {
          0%,
          100% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
        }

        .active-users-indicator:hover {
          background: rgba(139, 90, 60, 0.15);
          border-color: rgba(139, 90, 60, 0.5);
          transform: scale(1.02);
        }

        @media (max-width: 768px) {
          .active-users-indicator {
            padding: 4px 8px;
            font-size: 12px;
          }
        }
      `}</style>
    </div>
  );
}
