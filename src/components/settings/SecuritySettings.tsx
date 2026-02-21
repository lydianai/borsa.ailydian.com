'use client';

import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';
// TwoFactorAuth temporarily disabled due to compilation issues
// import TwoFactorAuth from './TwoFactorAuth';

export default function SecuritySettings({ onSave }: any) {
  const [settings, setSettings] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success'>('idle');
  const [activeTab, setActiveTab] = useState(1);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/security-settings');
      const data = await response.json();
      if (data.success) {
        setSettings(data.data.settings);
        setStats(data.data.stats);
      }
    } finally {
      setLoading(false);
    }
  };

  const updateSettings = async (updatedData: any) => {
    setSaveStatus('saving');
    try {
      const response = await fetch('/api/security-settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedData),
      });
      const data = await response.json();
      if (data.success) {
        setSettings(data.data);
        setSaveStatus('success');
        onSave?.();
        setTimeout(() => setSaveStatus('idle'), 2000);
      }
    } catch (error) {
      console.error('Save error:', error);
    }
  };

  const addCurrentIPToWhitelist = async () => {
    const response = await fetch('/api/security-settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add_ip_to_whitelist' }),
    });
    if (response.ok) {
      fetchSettings();
    }
  };

  const clearLoginHistory = async () => {
    if (!confirm('Tüm giriş geçmişi silinecek. Devam edilsin mi?')) return;
    const response = await fetch('/api/security-settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'clear_login_history' }),
    });
    if (response.ok) {
      alert('✅ Giriş geçmişi temizlendi');
      fetchSettings();
    }
  };

  if (loading) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-loading">
          <Icons.Loader2 className="animate-spin w-8 h-8" />
        </div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">Güvenlik ayarları yüklenemedi</div>
      </div>
    );
  }

  return (
    <div className="settings-content-wrapper">
      {/* Stats Cards */}
      <div className="settings-grid-3 mb-6">
        <div className="settings-stat-card">
          <div className="stat-icon"><Icons.Lock style={{ color: '#EF4444' }} className="w-6 h-6" /></div>
          <div className="stat-value">{stats?.twoFactorEnabled ? 'AÇIK' : 'KAPALI'}</div>
          <div className="stat-label">2FA Durumu</div>
        </div>
        <div className="settings-stat-card">
          <div className="stat-icon"><Icons.Monitor style={{ color: '#EF4444' }} className="w-6 h-6" /></div>
          <div className="stat-value">{stats?.activeSessions || 0}</div>
          <div className="stat-label">Aktif Oturumlar</div>
        </div>
        <div className="settings-stat-card">
          <div className="stat-icon"><Icons.BarChart3 style={{ color: '#EF4444' }} className="w-8 h-8" /></div>
          <div className="stat-value">{stats?.totalLogins || 0}</div>
          <div className="stat-label">Toplam Giriş</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 overflow-x-auto">
        {[
          { id: 1, label: '2FA' },
          { id: 2, label: 'IP Beyaz Liste' },
          { id: 3, label: 'Oturumlar' },
          { id: 4, label: 'Giriş Geçmişi' },
          { id: 5, label: 'API Güvenliği' },
        ].map((tab) => (
          <button
            key={tab.id}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              activeTab === tab.id
                ? 'bg-black text-white border-2 border-white'
                : 'bg-black text-gray-400 hover:bg-black border border-white/30'
            }`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: 2FA - Google Authenticator */}
      {activeTab === 1 && (
        <div className="settings-premium-card">
          <div style={{ textAlign: 'center', padding: '48px 24px' }}>
            <Icons.Settings className="w-16 h-16" style={{ color: '#9CA3AF', marginBottom: '16px' }} />
            <h3 style={{ fontSize: '20px', fontWeight: '600', color: '#FFFFFF', marginBottom: '8px' }}>
              Yapım Aşamasında
            </h3>
            <p style={{ color: '#9CA3AF', fontSize: '14px' }}>
              İki faktörlü kimlik doğrulama özelliği yakında eklenecek.
            </p>
          </div>
        </div>
      )}

      {/* Tab 2: IP Whitelist */}
      {activeTab === 2 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon"><Icons.Globe style={{ color: '#EF4444' }} className="w-6 h-6" /></div>
            <h3>IP Beyaz Liste</h3>
            <label className="settings-toggle-premium ml-auto">
              <input
                type="checkbox"
                checked={settings.ipWhitelist.enabled}
                onChange={(e) => updateSettings({ ipWhitelist: { ...settings.ipWhitelist, enabled: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="settings-card-body">
            <div className="settings-alert-info mb-4">
              <div className="text-sm text-gray-400">Şu Anki IP Adresiniz</div>
              <div className="text-lg font-bold text-cyan-400">{settings.ipWhitelist.currentIP}</div>
              <button
                onClick={addCurrentIPToWhitelist}
                style={{
                  marginTop: '12px',
                  padding: '10px 20px',
                  background: 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '8px',
                  fontWeight: '600',
                  fontSize: '14px',
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  boxShadow: '0 2px 8px rgba(59, 130, 246, 0.3)',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-2px)';
                  e.currentTarget.style.boxShadow = '0 4px 16px rgba(59, 130, 246, 0.5)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(59, 130, 246, 0.3)';
                }}
              >
                Bu IP'yi Whitelist'e Ekle
              </button>
            </div>

            <div className="settings-form-group">
              <label>İzin Verilen IP Adresleri</label>
              {settings.ipWhitelist.allowedIPs.length === 0 ? (
                <div className="text-center py-8 text-gray-500">Henüz IP eklenmemiş</div>
              ) : (
                <div className="space-y-2">
                  {settings.ipWhitelist.allowedIPs.map((ip: string, index: number) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                      <span className="font-mono">{ip}</span>
                      <button className="text-red-400 hover:text-red-300">
                        <Icons.Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={settings.ipWhitelist.blockUnknown}
                onChange={(e) => updateSettings({ ipWhitelist: { ...settings.ipWhitelist, blockUnknown: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Bilinmeyen IP'leri engelle</span>
            </label>
          </div>
        </div>
      )}

      {/* Tab 3: Session Management */}
      {activeTab === 3 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon"><Icons.Monitor style={{ color: '#EF4444' }} className="w-6 h-6" /></div>
            <h3>Oturum Yönetimi</h3>
          </div>

          <div className="settings-card-body">
            <div className="settings-form-group">
              <label>Maksimum Aktif Oturum</label>
              <input
                type="number"
                className="settings-input-premium"
                value={settings.sessionManagement.maxActiveSessions}
                onChange={(e) => updateSettings({ sessionManagement: { ...settings.sessionManagement, maxActiveSessions: Number(e.target.value) } })}
                min="1"
                max="10"
              />
            </div>

            <div className="settings-grid-2">
              <div className="settings-form-group">
                <label>Oturum Zaman Aşımı (dakika)</label>
                <input
                  type="number"
                  className="settings-input-premium"
                  value={settings.sessionManagement.sessionTimeout}
                  onChange={(e) => updateSettings({ sessionManagement: { ...settings.sessionManagement, sessionTimeout: Number(e.target.value) } })}
                  min="5"
                  max="1440"
                />
              </div>
              <div className="settings-form-group">
                <label>İnaktivite Zaman Aşımı (dakika)</label>
                <input
                  type="number"
                  className="settings-input-premium"
                  value={settings.sessionManagement.inactivityTimeout}
                  onChange={(e) => updateSettings({ sessionManagement: { ...settings.sessionManagement, inactivityTimeout: Number(e.target.value) } })}
                  min="5"
                  max="120"
                />
              </div>
            </div>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={settings.sessionManagement.autoLogoutOnInactive}
                onChange={(e) => updateSettings({ sessionManagement: { ...settings.sessionManagement, autoLogoutOnInactive: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">İnaktivitede otomatik çıkış</span>
            </label>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={settings.sessionManagement.requireReauthForSensitive}
                onChange={(e) => updateSettings({ sessionManagement: { ...settings.sessionManagement, requireReauthForSensitive: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Hassas işlemlerde tekrar kimlik doğrula</span>
            </label>
          </div>
        </div>
      )}

      {/* Tab 4: Login History */}
      {activeTab === 4 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon"><Icons.ScrollText style={{ color: '#EF4444' }} className="w-6 h-6" /></div>
            <h3>Giriş Geçmişi</h3>
            <button
              onClick={clearLoginHistory}
              style={{
                marginLeft: 'auto',
                padding: '8px 16px',
                background: 'linear-gradient(135deg, #EF4444, #DC2626)',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                fontWeight: '600',
                fontSize: '13px',
                cursor: 'pointer',
                transition: 'all 0.3s',
                boxShadow: '0 2px 6px rgba(239, 68, 68, 0.3)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(239, 68, 68, 0.5)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 6px rgba(239, 68, 68, 0.3)';
              }}
            >
              Geçmişi Temizle
            </button>
          </div>

          <div className="settings-card-body">
            <div className="settings-grid-2 mb-4">
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={settings.loginHistory.notifyOnNewDevice}
                  onChange={(e) => updateSettings({ loginHistory: { ...settings.loginHistory, notifyOnNewDevice: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">Yeni cihazda bildir</span>
              </label>
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={settings.loginHistory.notifyOnUnusualLocation}
                  onChange={(e) => updateSettings({ loginHistory: { ...settings.loginHistory, notifyOnUnusualLocation: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">Farklı konumda bildir</span>
              </label>
            </div>

            <div className="space-y-3 max-h-96 overflow-y-auto">
              {settings.loginHistory.history.map((entry: any, i: number) => (
                <div key={i} className={`p-4 rounded-lg border ${entry.success ? 'bg-gray-900/50 border-gray-700' : 'bg-red-600/10 border-red-600/30'}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {entry.success ? <Icons.CheckCircle2 className="text-green-400 w-5 h-5" /> : <Icons.XCircle className="text-red-400 w-5 h-5" />}
                      <span className="font-semibold">{entry.success ? 'Başarılı Giriş' : 'Başarısız Giriş'}</span>
                    </div>
                    <span className="text-xs text-gray-400">
                      {new Date(entry.timestamp).toLocaleString('tr-TR')}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm text-gray-400">
                    <div>
                      <Icons.MapPin className="inline mr-1 w-3.5 h-3.5" />
                      {entry.location}
                    </div>
                    <div>
                      <Icons.Globe className="inline mr-1 w-3.5 h-3.5" />
                      {entry.ip}
                    </div>
                    <div>
                      <Icons.Monitor className="inline mr-1 w-3.5 h-3.5" />
                      {entry.device}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab 5: API Security */}
      {activeTab === 5 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon"><Icons.Lock style={{ color: '#EF4444' }} className="w-6 h-6" /></div>
            <h3>API Güvenliği</h3>
          </div>

          <div className="settings-card-body">
            <div className="flex justify-between items-center p-4 bg-gray-900/50 rounded-lg mb-4">
              <div>
                <div className="font-semibold">Rate Limiting</div>
                <div className="text-sm text-gray-400">API çağrı hızını sınırla</div>
              </div>
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={settings.apiSecurity.rateLimiting.enabled}
                  onChange={(e) =>
                    updateSettings({
                      apiSecurity: {
                        ...settings.apiSecurity,
                        rateLimiting: { ...settings.apiSecurity.rateLimiting, enabled: e.target.checked },
                      },
                    })
                  }
                />
                <span className="toggle-slider"></span>
              </label>
            </div>

            <div className="settings-grid-2">
              <div className="settings-form-group">
                <label>Max İstek / Dakika</label>
                <input
                  type="number"
                  className="settings-input-premium"
                  value={settings.apiSecurity.rateLimiting.maxRequests}
                  onChange={(e) =>
                    updateSettings({
                      apiSecurity: {
                        ...settings.apiSecurity,
                        rateLimiting: { ...settings.apiSecurity.rateLimiting, maxRequests: Number(e.target.value) },
                      },
                    })
                  }
                  min="10"
                  max="10000"
                />
              </div>
              <div className="settings-form-group">
                <label>Burst Limit</label>
                <input
                  type="number"
                  className="settings-input-premium"
                  value={settings.apiSecurity.rateLimiting.burstLimit}
                  onChange={(e) =>
                    updateSettings({
                      apiSecurity: {
                        ...settings.apiSecurity,
                        rateLimiting: { ...settings.apiSecurity.rateLimiting, burstLimit: Number(e.target.value) },
                      },
                    })
                  }
                  min="10"
                  max="10000"
                />
              </div>
            </div>

            <div className="settings-form-group">
              <label>İzin Verilen Origin'ler</label>
              <div className="space-y-2">
                {settings.apiSecurity.allowedOrigins.map((origin: string, i: number) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-gray-900/50 rounded-lg">
                    <span className="font-mono text-sm">{origin}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Save Status */}
      {saveStatus === 'saving' && (
        <div className="settings-alert-info">
          <Icons.Loader2 className="animate-spin w-4 h-4" />
          <span>Kaydediliyor...</span>
        </div>
      )}
      {saveStatus === 'success' && (
        <div className="settings-alert-success">
          <Icons.Check className="w-4 h-4" />
          <span>✅ Kaydedildi!</span>
        </div>
      )}
    </div>
  );
}
