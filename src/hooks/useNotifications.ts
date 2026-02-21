/**
 * USE NOTIFICATIONS HOOK
 * React hook for real-time notifications with auto-polling
 */

'use client';

import { useState, useEffect, useCallback } from 'react';
import { NotificationStore, type Notification } from '@/lib/notifications/NotificationStore';

export function useNotifications(autoFetch = true, fetchInterval = 10000) {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [loading, setLoading] = useState(false);

  // Update state from store
  const updateState = useCallback(() => {
    setNotifications(NotificationStore.getAll());
    setUnreadCount(NotificationStore.getUnreadCount());
  }, []);

  // Fetch notifications from API
  const fetchNotifications = useCallback(async () => {
    if (!autoFetch) return;

    try {
      setLoading(true);
      const response = await fetch('/api/notifications');
      const result = await response.json();

      if (result.success && result.data) {
        // Add each notification to store
        result.data.forEach((notif: any) => {
          // Check if notification already exists
          const existing = NotificationStore.getAll().find(
            n => n.data && n.data.symbol === notif.data?.symbol && n.type === notif.type
          );

          if (!existing) {
            NotificationStore.add(notif);
          }
        });
      }
    } catch (err) {
      console.error('[useNotifications] Fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [autoFetch]);

  // Subscribe to notifications
  useEffect(() => {
    // Initial load
    updateState();

    // Subscribe to changes
    const unsubscribe = NotificationStore.subscribe(() => {
      updateState();
    });

    return unsubscribe;
  }, [updateState]);

  // Auto-fetch from API
  useEffect(() => {
    if (!autoFetch) return;

    // Fetch immediately
    fetchNotifications();

    // Then poll at interval
    const interval = setInterval(fetchNotifications, fetchInterval);

    return () => clearInterval(interval);
  }, [autoFetch, fetchInterval, fetchNotifications]);

  // Add notification
  const addNotification = useCallback((notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    return NotificationStore.add(notification);
  }, []);

  // Mark as read
  const markAsRead = useCallback((id: string) => {
    NotificationStore.markAsRead(id);
  }, []);

  // Mark all as read
  const markAllAsRead = useCallback(() => {
    NotificationStore.markAllAsRead();
  }, []);

  // Delete notification
  const deleteNotification = useCallback((id: string) => {
    NotificationStore.delete(id);
  }, []);

  // Clear all
  const clearAll = useCallback(() => {
    NotificationStore.clearAll();
  }, []);

  return {
    notifications,
    unreadCount,
    loading,
    addNotification,
    markAsRead,
    markAllAsRead,
    deleteNotification,
    clearAll,
    refresh: fetchNotifications
  };
}
