'use client';

/**
 * Admin Panel Component
 *
 * White-hat compliance: Admin-only user management
 * Shows: Pending users, notifications, user approval
 */

import { useEffect, useState } from 'react';
import { Users, Bell, CheckCircle, XCircle, Loader2, AlertCircle } from 'lucide-react';

interface PendingUser {
  id: string;
  email: string;
  username: string;
  createdAt: string;
  emailVerified: boolean;
  hasActivePayment: boolean;
  subscriptionTier: string;
}

interface AdminNotification {
  id: string;
  type: string;
  title: string;
  message: string;
  userEmail: string | null;
  isRead: boolean;
  createdAt: string;
  actionUrl: string | null;
}

export default function AdminPanel() {
  const [loading, setLoading] = useState(true);
  const [pendingUsers, setPendingUsers] = useState<PendingUser[]>([]);
  const [notifications, setNotifications] = useState<AdminNotification[]>([]);
  const [activeView, setActiveView] = useState<'users' | 'notifications'>('users');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      // Load pending users
      const usersRes = await fetch('/api/admin/users?status=pending');
      if (usersRes.ok) {
        const usersData = await usersRes.json();
        setPendingUsers(usersData.users || []);
      }

      // Load notifications
      const notifsRes = await fetch('/api/admin/notifications?unreadOnly=true');
      if (notifsRes.ok) {
        const notifsData = await notifsRes.json();
        setNotifications(notifsData.notifications || []);
      }
    } catch (error) {
      console.error('Admin data load error:', error);
    } finally {
      setLoading(false);
    }
  };

  const approveUser = async (userId: string) => {
    try {
      const response = await fetch(`/api/admin/users/${userId}/approve`, {
        method: 'POST',
      });

      const data = await response.json();

      if (response.ok) {
        alert('✅ Kullanıcı başarıyla onaylandı! Email gönderildi.');
        loadData(); // Reload data
      } else {
        alert('❌ ' + (data.error || 'Onaylama başarısız'));
      }
    } catch (error) {
      alert('❌ Bir hata oluştu');
    }
  };

  const markAsRead = async (notificationId: string) => {
    try {
      await fetch('/api/admin/notifications', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notificationId }),
      });
      loadData();
    } catch (error) {
      console.error('Mark as read error:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-red-400">🔴 Admin Panel</h2>
          <p className="text-gray-400 text-sm">Yönetici özellikleri</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setActiveView('users')}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              activeView === 'users'
                ? 'bg-red-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <Users className="w-4 h-4 inline mr-2" />
            Kullanıcılar ({pendingUsers.length})
          </button>
          <button
            onClick={() => setActiveView('notifications')}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              activeView === 'notifications'
                ? 'bg-red-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <Bell className="w-4 h-4 inline mr-2" />
            Bildirimler ({notifications.length})
          </button>
        </div>
      </div>

      {/* Pending Users View */}
      {activeView === 'users' && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <AlertCircle className="w-5 h-5 text-yellow-500" />
            <h3 className="text-lg font-bold">
              Onay Bekleyen Kullanıcılar ({pendingUsers.length})
            </h3>
          </div>

          {pendingUsers.length === 0 ? (
            <div className="text-center py-12 bg-gray-800/50 rounded-lg border border-gray-700">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
              <p className="text-gray-400">Onay bekleyen kullanıcı yok</p>
            </div>
          ) : (
            <div className="space-y-3">
              {pendingUsers.map((user) => (
                <div
                  key={user.id}
                  className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-red-500/50 transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h4 className="font-bold text-lg">{user.username}</h4>
                        {user.emailVerified ? (
                          <span className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded border border-green-500/30">
                            ✓ Email Verified
                          </span>
                        ) : (
                          <span className="text-xs bg-yellow-900/30 text-yellow-400 px-2 py-1 rounded border border-yellow-500/30">
                            ⏳ Email Pending
                          </span>
                        )}
                      </div>
                      <p className="text-gray-400 text-sm mb-1">📧 {user.email}</p>
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>
                          📅 Kayıt: {new Date(user.createdAt).toLocaleDateString('tr-TR')}
                        </span>
                        <span>💳 Plan: {user.subscriptionTier.toUpperCase()}</span>
                        {user.hasActivePayment && (
                          <span className="text-green-400">✓ Ödeme Var</span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => approveUser(user.id)}
                      disabled={!user.emailVerified}
                      className={`px-6 py-3 rounded-lg font-bold transition-all ${
                        user.emailVerified
                          ? 'bg-green-600 hover:bg-green-700 text-white'
                          : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                      }`}
                    >
                      {user.emailVerified ? '✓ Onayla' : '⏳ Email Bekliyor'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Notifications View */}
      {activeView === 'notifications' && (
        <div>
          <div className="flex items-center gap-2 mb-4">
            <Bell className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-bold">
              Okunmamış Bildirimler ({notifications.length})
            </h3>
          </div>

          {notifications.length === 0 ? (
            <div className="text-center py-12 bg-gray-800/50 rounded-lg border border-gray-700">
              <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-3" />
              <p className="text-gray-400">Yeni bildirim yok</p>
            </div>
          ) : (
            <div className="space-y-3">
              {notifications.map((notif) => (
                <div
                  key={notif.id}
                  className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-blue-500/50 transition-all"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-bold mb-1">{notif.title}</h4>
                      <p className="text-gray-400 text-sm mb-2">{notif.message}</p>
                      {notif.userEmail && (
                        <p className="text-xs text-gray-500">📧 {notif.userEmail}</p>
                      )}
                      <p className="text-xs text-gray-600 mt-2">
                        {new Date(notif.createdAt).toLocaleString('tr-TR')}
                      </p>
                    </div>
                    <button
                      onClick={() => markAsRead(notif.id)}
                      className="text-xs bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded transition-all"
                    >
                      Okundu İşaretle
                    </button>
                  </div>
                  {notif.actionUrl && (
                    <a
                      href={notif.actionUrl}
                      className="inline-block mt-3 text-sm text-blue-400 hover:text-blue-300"
                    >
                      → İncele
                    </a>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 mt-6">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Bekleyen Onay</p>
          <p className="text-3xl font-bold text-yellow-400">{pendingUsers.length}</p>
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
          <p className="text-gray-400 text-sm mb-1">Okunmamış Bildirim</p>
          <p className="text-3xl font-bold text-blue-400">{notifications.length}</p>
        </div>
      </div>
    </div>
  );
}
