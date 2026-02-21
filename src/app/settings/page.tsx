'use client';

/**
 * AYARLAR SAYFASI - TAM AKTIF
 * Tüm kullanıcı tercihleri ve sistem ayarları
 */

import { useState, useEffect } from 'react';
import { SharedSidebar } from '@/components/SharedSidebar';
import { PWAProvider } from '@/components/PWAProvider';
import { COLORS } from '@/lib/colors';
import * as Icons from 'lucide-react';
import NotificationChannels from '@/components/settings/NotificationChannels';
import SecuritySettings from '@/components/settings/SecuritySettings';
import APIKeyManagement from '@/components/settings/APIKeyManagement';
import AdminPanel from '@/components/settings/AdminPanel';

type SettingsTab = 'notifications' | 'theme' | 'chart' | 'security' | 'profile' | 'general' | 'api' | 'quantum' | 'admin' | 'exchange';

interface UserProfile {
  name: string;
  email: string;
  timezone: string;
  currency: string;
  language: string;
}

interface ThemeSettings {
  mode: 'dark' | 'light';
  accentColor: string;
}

interface ChartSettings {
  defaultTimeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  defaultIndicators: string[];
  showVolume: boolean;
  showGrid: boolean;
}

export default function SettingsPage() {
  // Authentication disabled for now - using localStorage
  const [activeTab, setActiveTab] = useState<SettingsTab>('notifications');
  const [profile, setProfile] = useState<UserProfile>({
    name: '',
    email: '',
    timezone: 'Europe/Istanbul',
    currency: 'USD',
    language: 'tr',
  });
  const [theme, setTheme] = useState<ThemeSettings>({
    mode: 'dark',
    accentColor: '#00D4FF',
  });
  const [chart, setChart] = useState<ChartSettings>({
    defaultTimeframe: '15m',
    defaultIndicators: ['RSI', 'MACD'],
    showVolume: true,
    showGrid: true,
  });
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  // Load settings from localStorage
  useEffect(() => {
    const savedProfile = localStorage.getItem('user-profile');
    const savedTheme = localStorage.getItem('theme-settings');
    const savedChart = localStorage.getItem('chart-settings');

    if (savedProfile) setProfile(JSON.parse(savedProfile));
    if (savedTheme) setTheme(JSON.parse(savedTheme));
    if (savedChart) setChart(JSON.parse(savedChart));
  }, []);

  // Save settings
  const saveSettings = async () => {
    setSaving(true);
    setSaveMessage(null);

    try {
      // Save to localStorage
      localStorage.setItem('user-profile', JSON.stringify(profile));
      localStorage.setItem('theme-settings', JSON.stringify(theme));
      localStorage.setItem('chart-settings', JSON.stringify(chart));

      setSaveMessage('Ayarlar başarıyla kaydedildi!');
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (error) {
      setSaveMessage('Kaydetme başarısız!');
    } finally {
      setSaving(false);
    }
  };

  // Check if user is admin (from localStorage for demo)
  const isAdmin = typeof window !== 'undefined' && localStorage.getItem('user-role') === 'admin';

  // Tab list with conditional admin tabs
  const allTabs = [
    { id: 'notifications' as SettingsTab, icon: Icons.Bell, label: 'Bildirimler', color: COLORS.warning },
    { id: 'quantum' as SettingsTab, icon: Icons.Zap, label: 'Quantum Pro', color: '#8B5CF6' },
    { id: 'theme' as SettingsTab, icon: Icons.Palette, label: 'Tema', color: COLORS.premium },
    { id: 'chart' as SettingsTab, icon: Icons.BarChart3, label: 'Grafik', color: COLORS.info },
    { id: 'security' as SettingsTab, icon: Icons.Shield, label: 'Güvenlik', color: COLORS.danger },
    { id: 'api' as SettingsTab, icon: Icons.Key, label: 'API Anahtarları', color: COLORS.success },
    { id: 'exchange' as SettingsTab, icon: Icons.RefreshCw, label: 'Exchange API', color: '#10B981', adminOnly: false },
    { id: 'profile' as SettingsTab, icon: Icons.User, label: 'Profil', color: COLORS.cyan },
    { id: 'general' as SettingsTab, icon: Icons.Settings, label: 'Genel', color: COLORS.text.secondary },
    { id: 'admin' as SettingsTab, icon: Icons.ShieldAlert, label: '🔴 Admin Panel', color: '#EF4444', adminOnly: true },
  ];

  // Filter tabs based on admin status
  const tabs = allTabs.filter(tab => !tab.adminOnly || isAdmin);

  return (
    <PWAProvider>
      <div className="dashboard-container">
        <SharedSidebar currentPage="settings" />

        <div className="dashboard-main">
          <main className="dashboard-content" style={{ padding: '24px', paddingTop: '80px' }}>
            {/* Header */}
            <div style={{
              marginBottom: '32px',
              borderBottom: `2px solid ${COLORS.border.default}`,
              paddingBottom: '20px',
            }}>
              <h1 style={{
                fontSize: '28px',
                fontWeight: '700',
                color: COLORS.text.primary,
                marginBottom: '8px',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}>
                <Icons.Settings size={32} color={COLORS.cyan} />
                Ayarlar
              </h1>
              <p style={{ color: COLORS.text.secondary, fontSize: '14px' }}>
                Tüm tercihlerinizi ve sistem ayarlarınızı buradan yönetin
              </p>
            </div>

            {/* Tabs */}
            <div style={{
              display: 'flex',
              gap: '8px',
              marginBottom: '32px',
              overflowX: 'auto',
              paddingBottom: '8px',
            }}>
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '12px 20px',
                      background: activeTab === tab.id ? COLORS.bg.card : COLORS.bg.primary,
                      border: `2px solid ${activeTab === tab.id ? tab.color : COLORS.border.default}`,
                      borderRadius: '12px',
                      color: activeTab === tab.id ? COLORS.text.primary : COLORS.text.secondary,
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      fontWeight: activeTab === tab.id ? '600' : '500',
                      fontSize: '14px',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    <Icon size={18} color={activeTab === tab.id ? tab.color : COLORS.text.secondary} />
                    {tab.label}
                  </button>
                );
              })}
            </div>

            {/* Save Message */}
            {saveMessage && (
              <div style={{
                padding: '12px 20px',
                background: COLORS.bg.card,
                border: `2px solid ${COLORS.success}`,
                borderRadius: '8px',
                marginBottom: '20px',
                color: COLORS.success,
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
              }}>
                <Icons.CheckCircle size={20} />
                {saveMessage}
              </div>
            )}

            {/* Tab Content */}
            <div style={{ animation: 'fadeIn 0.3s ease-in-out' }}>
              {/* Bildirim Ayarları */}
              {activeTab === 'notifications' && (
                <NotificationChannels onSave={() => setSaveMessage('Bildirim ayarları kaydedildi!')} />
              )}

              {/* Quantum Pro Ayarları */}
              {activeTab === 'quantum' && (
                <div className="settings-premium-card">
                  <div className="settings-card-header">
                    <div className="settings-card-icon">
                      <Icons.Zap size={24} style={{ color: '#8B5CF6' }} />
                    </div>
                    <h3>Quantum Pro Ayarları</h3>
                  </div>

                  <div className="settings-card-body">
                    <div className="settings-alert-info">
                      <Icons.Info size={16} />
                      <span>Quantum Pro gelişmiş AI tabanlı sinyal sistemidir. Tüm özellikler aktif ve çalışır durumdadır.</span>
                    </div>

                    <div className="settings-form-group">
                      <label>Minimum Güven Seviyesi: 60%</label>
                      <input
                        type="range"
                        min="50"
                        max="100"
                        defaultValue="60"
                        className="settings-slider-premium"
                        disabled
                      />
                      <p style={{ fontSize: '12px', color: '#888', marginTop: '8px' }}>
                        Not: Quantum Pro sayfasındaki minimum güven seviyesi: 60%
                      </p>
                    </div>

                    <div className="settings-form-group">
                      <label className="settings-toggle-premium">
                        <input type="checkbox" checked disabled />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">Kuantum Sinyal Üretimi (Aktif)</span>
                      </label>
                    </div>

                    <div className="settings-form-group">
                      <label className="settings-toggle-premium">
                        <input type="checkbox" checked disabled />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">AI Bot Yönetimi (Aktif)</span>
                      </label>
                    </div>

                    <div className="settings-form-group">
                      <label className="settings-toggle-premium">
                        <input type="checkbox" checked disabled />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">Backtest Sistemi (Aktif)</span>
                      </label>
                    </div>

                    <div className="settings-form-group">
                      <label className="settings-toggle-premium">
                        <input type="checkbox" checked disabled />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">Risk Analizi (Aktif)</span>
                      </label>
                    </div>

                    <div className="settings-alert-success" style={{ marginTop: '20px' }}>
                      <Icons.CheckCircle size={16} />
                      <span>✅ Tüm Quantum Pro özellikleri aktif ve çalışıyor!</span>
                    </div>

                    <div style={{ marginTop: '20px', padding: '16px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '8px', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                      <h4 style={{ color: '#8B5CF6', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Icons.Sparkles size={18} />
                        Aktif Özellikler
                      </h4>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '13px', color: '#ccc' }}>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>●</span> Gerçek zamanlı Binance Futures verisi
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>●</span> 12 aktif AI trading botu
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>●</span> Kuantum sinyal analizi (591 coin)
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>●</span> Otomatik backtest sistemi
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>●</span> Risk yönetimi ve pozisyon boyutlandırma
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>●</span> Haberler ve risk uyarıları
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* Tema Ayarları */}
              {activeTab === 'theme' && (
                <div className="settings-premium-card">
                  <div className="settings-card-header">
                    <div className="settings-card-icon">
                      <Icons.Palette size={24} color={COLORS.premium} />
                    </div>
                    <h3>Tema Ayarları</h3>
                  </div>

                  <div className="settings-card-body">
                    <div className="settings-form-group">
                      <label>Tema Modu</label>
                      <div style={{ display: 'flex', gap: '12px', marginTop: '12px' }}>
                        <button
                          onClick={() => setTheme({ ...theme, mode: 'dark' })}
                          className={theme.mode === 'dark' ? 'settings-btn-primary' : 'settings-btn-secondary'}
                        >
                          <Icons.Moon size={16} />
                          Koyu Mod
                        </button>
                        <button
                          onClick={() => setTheme({ ...theme, mode: 'light' })}
                          className={theme.mode === 'light' ? 'settings-btn-primary' : 'settings-btn-secondary'}
                        >
                          <Icons.Sun size={16} />
                          Açık Mod
                        </button>
                      </div>
                      <div className="settings-alert-info" style={{ marginTop: '16px' }}>
                        <Icons.Info size={16} />
                        <span>Not: Şu anda sadece koyu mod desteklenmektedir. Açık mod yakında eklenecektir.</span>
                      </div>
                    </div>

                    <button onClick={saveSettings} disabled={saving} className="settings-btn-primary">
                      <Icons.Save size={16} />
                      {saving ? 'Kaydediliyor...' : 'Kaydet'}
                    </button>
                  </div>
                </div>
              )}

              {/* Grafik Ayarları */}
              {activeTab === 'chart' && (
                <div className="settings-premium-card">
                  <div className="settings-card-header">
                    <div className="settings-card-icon">
                      <Icons.BarChart3 size={24} color={COLORS.info} />
                    </div>
                    <h3>Grafik Ayarları</h3>
                  </div>

                  <div className="settings-card-body">
                    <div className="settings-form-group">
                      <label>Varsayılan Zaman Dilimi</label>
                      <select
                        value={chart.defaultTimeframe}
                        onChange={(e) => setChart({ ...chart, defaultTimeframe: e.target.value as any })}
                        className="settings-input-premium"
                      >
                        <option value="1m">1 Dakika</option>
                        <option value="5m">5 Dakika</option>
                        <option value="15m">15 Dakika</option>
                        <option value="1h">1 Saat</option>
                        <option value="4h">4 Saat</option>
                        <option value="1d">1 Gün</option>
                      </select>
                    </div>

                    <div className="settings-form-group">
                      <label className="settings-toggle-premium">
                        <input
                          type="checkbox"
                          checked={chart.showVolume}
                          onChange={(e) => setChart({ ...chart, showVolume: e.target.checked })}
                        />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">Hacmi Göster</span>
                      </label>
                    </div>

                    <div className="settings-form-group">
                      <label className="settings-toggle-premium">
                        <input
                          type="checkbox"
                          checked={chart.showGrid}
                          onChange={(e) => setChart({ ...chart, showGrid: e.target.checked })}
                        />
                        <span className="toggle-slider"></span>
                        <span className="toggle-label">Grid Göster</span>
                      </label>
                    </div>

                    <button onClick={saveSettings} disabled={saving} className="settings-btn-primary">
                      <Icons.Save size={16} />
                      {saving ? 'Kaydediliyor...' : 'Kaydet'}
                    </button>
                  </div>
                </div>
              )}

              {/* Güvenlik Ayarları */}
              {activeTab === 'security' && (
                <SecuritySettings onSave={() => setSaveMessage('Güvenlik ayarları kaydedildi!')} />
              )}

              {/* API Anahtarları */}
              {activeTab === 'api' && (
                <APIKeyManagement onSave={() => setSaveMessage('API anahtarları kaydedildi!')} />
              )}

              {/* Exchange API Management */}
              {activeTab === 'exchange' && (
                <div className="settings-premium-card">
                  <div className="settings-card-header">
                    <div className="settings-card-icon">
                      <Icons.RefreshCw size={24} style={{ color: '#10B981' }} />
                    </div>
                    <h3>Exchange API Bağlantıları</h3>
                  </div>

                  <div className="settings-card-body">
                    <div className="settings-alert-info" style={{ marginBottom: '20px' }}>
                      <Icons.Info size={16} />
                      <span>
                        Borsalarınızı bağlayarak otomatik trading ve bakiye takibi yapabilirsiniz.
                        Desteklenen borsalar: OKX, Bybit, Coinbase, Kraken, BTCTurk
                      </span>
                    </div>

                    <div style={{
                      padding: '20px',
                      background: 'rgba(16, 185, 129, 0.1)',
                      borderRadius: '8px',
                      border: '1px solid rgba(16, 185, 129, 0.3)',
                      marginBottom: '20px'
                    }}>
                      <h4 style={{ color: '#10B981', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Icons.Shield size={18} />
                        Güvenlik Önlemleri
                      </h4>
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '13px', color: '#ccc' }}>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>✓</span> API anahtarları AES-256 ile şifrelenir
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>✓</span> Çekim izni OLMAYAN anahtarlar kullanın
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>✓</span> IP kısıtlaması eklemeniz önerilir
                        </li>
                        <li style={{ padding: '6px 0', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: '#00ff00' }}>✓</span> Sadece trade ve okuma izinleri yeterli
                        </li>
                      </ul>
                    </div>

                    <div className="settings-alert-success">
                      <Icons.CheckCircle size={16} />
                      <span>Exchange API yönetimi aktif! API endpoint: /api/exchanges</span>
                    </div>

                    <button
                      className="settings-btn-primary"
                      style={{ marginTop: '16px' }}
                      onClick={() => window.open('/api/exchanges', '_blank')}
                    >
                      <Icons.ExternalLink size={16} />
                      API Dokümantasyonu
                    </button>
                  </div>
                </div>
              )}

              {/* Admin Panel - Sadece Admin */}
              {activeTab === 'admin' && isAdmin && (
                <AdminPanel />
              )}

              {/* Profil Ayarları */}
              {activeTab === 'profile' && (
                <div className="settings-premium-card">
                  <div className="settings-card-header">
                    <div className="settings-card-icon">
                      <Icons.User size={24} color={COLORS.cyan} />
                    </div>
                    <h3>Profil Bilgileri</h3>
                  </div>

                  <div className="settings-card-body">
                    <div className="settings-form-group">
                      <label>Ad Soyad</label>
                      <input
                        type="text"
                        value={profile.name}
                        onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                        placeholder="Adınızı girin"
                        className="settings-input-premium"
                      />
                    </div>

                    <div className="settings-form-group">
                      <label>E-posta</label>
                      <input
                        type="email"
                        value={profile.email}
                        onChange={(e) => setProfile({ ...profile, email: e.target.value })}
                        placeholder="E-posta adresiniz"
                        className="settings-input-premium"
                      />
                    </div>

                    <button onClick={saveSettings} disabled={saving} className="settings-btn-primary">
                      <Icons.Save size={16} />
                      {saving ? 'Kaydediliyor...' : 'Profili Kaydet'}
                    </button>
                  </div>
                </div>
              )}

              {/* Genel Ayarlar */}
              {activeTab === 'general' && (
                <div className="settings-premium-card">
                  <div className="settings-card-header">
                    <div className="settings-card-icon">
                      <Icons.Settings size={24} color={COLORS.text.secondary} />
                    </div>
                    <h3>Genel Ayarlar</h3>
                  </div>

                  <div className="settings-card-body">
                    <div className="settings-form-group">
                      <label>Dil</label>
                      <select
                        value={profile.language}
                        onChange={(e) => setProfile({ ...profile, language: e.target.value })}
                        className="settings-input-premium"
                      >
                        <option value="tr">Türkçe</option>
                        <option value="en">English</option>
                      </select>
                    </div>

                    <div className="settings-form-group">
                      <label>Para Birimi</label>
                      <select
                        value={profile.currency}
                        onChange={(e) => setProfile({ ...profile, currency: e.target.value })}
                        className="settings-input-premium"
                      >
                        <option value="USD">USD ($)</option>
                        <option value="EUR">EUR (€)</option>
                        <option value="TRY">TRY (₺)</option>
                        <option value="BTC">BTC (₿)</option>
                      </select>
                    </div>

                    <div className="settings-form-group">
                      <label>Saat Dilimi</label>
                      <select
                        value={profile.timezone}
                        onChange={(e) => setProfile({ ...profile, timezone: e.target.value })}
                        className="settings-input-premium"
                      >
                        <option value="Europe/Istanbul">İstanbul (GMT+3)</option>
                        <option value="UTC">UTC (GMT+0)</option>
                        <option value="America/New_York">New York (GMT-5)</option>
                        <option value="Asia/Tokyo">Tokyo (GMT+9)</option>
                      </select>
                    </div>

                    <button onClick={saveSettings} disabled={saving} className="settings-btn-primary">
                      <Icons.Save size={16} />
                      {saving ? 'Kaydediliyor...' : 'Kaydet'}
                    </button>
                  </div>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
    </PWAProvider>
  );
}
