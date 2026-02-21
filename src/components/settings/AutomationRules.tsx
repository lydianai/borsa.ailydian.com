'use client';

import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';

export default function AutomationRules({ onSave }: any) {
  const [rules, setRules] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success'>('idle');
  const [activeTab, setActiveTab] = useState(1);

  useEffect(() => {
    fetchRules();
  }, []);

  const fetchRules = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/automation-rules');
      const data = await response.json();
      if (data.success) {
        setRules(data.data.rules);
        setStats(data.data.stats);
      }
    } finally {
      setLoading(false);
    }
  };

  const updateRules = async (updatedData: any) => {
    setSaveStatus('saving');
    try {
      const response = await fetch('/api/automation-rules', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedData),
      });
      const data = await response.json();
      if (data.success) {
        setRules(data.data);
        setSaveStatus('success');
        onSave?.();
        setTimeout(() => setSaveStatus('idle'), 2000);
      }
    } catch (error) {
      console.error('Save error:', error);
    }
  };

  const triggerManualBackup = async () => {
    const response = await fetch('/api/automation-rules', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'trigger_manual_backup' }),
    });
    const data = await response.json();
    if (data.success) {
      alert('✅ Yedekleme tamamlandı!');
      fetchRules();
    }
  };

  if (loading) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-loading">
          {(Icons.Loader2 as any)({ className: "animate-spin", size: 32 })}
        </div>
      </div>
    );
  }

  if (!rules) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">Otomasyon kuralları yüklenemedi</div>
      </div>
    );
  }

  return (
    <div className="settings-content-wrapper">
      {/* Stats Cards */}
      <div className="settings-grid-3 mb-6">
        <div className="settings-stat-card">
          <div className="stat-icon">{(Icons.Settings as any)({ style: { color: '#8B5CF6' }, size: 32 })}</div>
          <div className="stat-value">{stats?.activeRules || 0}</div>
          <div className="stat-label">Aktif Kurallar</div>
        </div>
        <div className="settings-stat-card">
          <div className="stat-icon">{(Icons.RefreshCw as any)({ style: { color: '#3B82F6' }, size: 32 })}</div>
          <div className="stat-value">{stats?.autoRefreshEnabled ? 'AÇIK' : 'KAPALI'}</div>
          <div className="stat-label">Otomatik Yenileme</div>
        </div>
        <div className="settings-stat-card">
          <div className="stat-icon">{(Icons.BarChart3 as any)({ style: { color: '#10B981' }, size: 32 })}</div>
          <div className="stat-value">{stats?.webhooksActive || 0}</div>
          <div className="stat-label">Webhooks</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 overflow-x-auto">
        {[
          { id: 1, label: 'Otomatik Yenileme' },
          { id: 2, label: 'Zamanlanmış Raporlar' },
          { id: 3, label: 'Uyarı Kuralları' },
          { id: 4, label: 'Veri Yedekleme' },
          { id: 5, label: 'Webhooks' },
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

      {/* Tab 1: Auto Refresh */}
      {activeTab === 1 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">{(Icons.RefreshCw as any)({ style: { color: '#3B82F6' }, size: 24 })}</div>
            <h3>Otomatik Yenileme</h3>
            <label className="settings-toggle-premium ml-auto">
              <input
                type="checkbox"
                checked={rules.autoRefresh.enabled}
                onChange={(e) => updateRules({ autoRefresh: { ...rules.autoRefresh, enabled: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="settings-card-body">
            <div className="settings-form-group">
              <label>Yenileme Aralığı (saniye)</label>
              <select
                className="settings-input-premium"
                value={rules.autoRefresh.interval}
                onChange={(e) => updateRules({ autoRefresh: { ...rules.autoRefresh, interval: Number(e.target.value) } })}
              >
                <option value="5">5 saniye</option>
                <option value="10">10 saniye</option>
                <option value="30">30 saniye</option>
                <option value="60">1 dakika</option>
                <option value="120">2 dakika</option>
              </select>
            </div>

            <div className="settings-form-group">
              <label>Otomatik Yenilenecek Sayfalar</label>
              <div className="space-y-2">
                {['trading-signals', 'ai-signals', 'quantum-signals', 'btc-eth-analysis'].map(page => (
                  <label key={page} className="settings-toggle-premium">
                    <input
                      type="checkbox"
                      checked={rules.autoRefresh.pages.includes(page)}
                      onChange={(e) => {
                        const newPages = e.target.checked
                          ? [...rules.autoRefresh.pages, page]
                          : rules.autoRefresh.pages.filter((p: string) => p !== page);
                        updateRules({ autoRefresh: { ...rules.autoRefresh, pages: newPages } });
                      }}
                    />
                    <span className="toggle-slider"></span>
                    <span className="toggle-label capitalize">{page.replace('-', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={rules.autoRefresh.pauseOnInactive}
                onChange={(e) => updateRules({ autoRefresh: { ...rules.autoRefresh, pauseOnInactive: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Tab aktif değilse duraklat</span>
            </label>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={rules.autoRefresh.showCountdown}
                onChange={(e) => updateRules({ autoRefresh: { ...rules.autoRefresh, showCountdown: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Geri sayımı göster</span>
            </label>
          </div>
        </div>
      )}

      {/* Tab 2: Scheduled Reports */}
      {activeTab === 2 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">{(Icons.BarChart3 as any)({ style: { color: '#10B981' }, size: 24 })}</div>
            <h3>Zamanlanmış Raporlar</h3>
            <label className="settings-toggle-premium ml-auto">
              <input
                type="checkbox"
                checked={rules.scheduledReports.enabled}
                onChange={(e) => updateRules({ scheduledReports: { ...rules.scheduledReports, enabled: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="settings-card-body">
            <div className="settings-grid-2">
              <div className="settings-form-group">
                <label>Sıklık</label>
                <select
                  className="settings-input-premium"
                  value={rules.scheduledReports.frequency}
                  onChange={(e) => updateRules({ scheduledReports: { ...rules.scheduledReports, frequency: e.target.value } })}
                >
                  <option value="hourly">Saatlik</option>
                  <option value="daily">Günlük</option>
                  <option value="weekly">Haftalık</option>
                  <option value="monthly">Aylık</option>
                </select>
              </div>

              <div className="settings-form-group">
                <label>Saat (HH:MM)</label>
                <input
                  type="time"
                  className="settings-input-premium"
                  value={rules.scheduledReports.time}
                  onChange={(e) => updateRules({ scheduledReports: { ...rules.scheduledReports, time: e.target.value } })}
                />
              </div>
            </div>

            <div className="settings-form-group">
              <label>Format</label>
              <select
                className="settings-input-premium"
                value={rules.scheduledReports.format}
                onChange={(e) => updateRules({ scheduledReports: { ...rules.scheduledReports, format: e.target.value } })}
              >
                <option value="PDF">PDF</option>
                <option value="HTML">HTML</option>
                <option value="JSON">JSON</option>
              </select>
            </div>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={rules.scheduledReports.includeCharts}
                onChange={(e) => updateRules({ scheduledReports: { ...rules.scheduledReports, includeCharts: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Grafikleri dahil et</span>
            </label>
          </div>
        </div>
      )}

      {/* Tab 3: Alert Rules */}
      {activeTab === 3 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">{(Icons.Bell as any)({ style: { color: '#F59E0B' }, size: 24 })}</div>
            <h3>Uyarı Kuralları</h3>
            <label className="settings-toggle-premium ml-auto">
              <input
                type="checkbox"
                checked={rules.alertAutomation.enabled}
                onChange={(e) => updateRules({ alertAutomation: { ...rules.alertAutomation, enabled: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="settings-card-body">
            <div className="space-y-4">
              {rules.alertAutomation.rules.map((rule: any) => (
                <div key={rule.id} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <div className="font-semibold text-lg">{rule.name}</div>
                      <div className="text-sm text-gray-400 mt-1">{rule.condition}</div>
                    </div>
                    <label className="settings-toggle-premium">
                      <input
                        type="checkbox"
                        checked={rule.active}
                        onChange={() => {
                          const updatedRules = rules.alertAutomation.rules.map((r: any) =>
                            r.id === rule.id ? { ...r, active: !r.active } : r
                          );
                          updateRules({ alertAutomation: { ...rules.alertAutomation, rules: updatedRules } });
                        }}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="text-gray-400">
                      Aksiyon: <span className="text-white font-semibold">{rule.action}</span>
                    </div>
                    <div className="text-gray-400">
                      Bekleme: <span className="text-white font-semibold">{rule.cooldown}d</span>
                    </div>
                  </div>
                  <div className="flex gap-2 mt-3">
                    {rule.channels.map((channel: string) => (
                      <span key={channel} className="px-3 py-1 bg-black text-white border border-white/50 rounded-full text-xs">
                        {channel}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Tab 4: Data Backup */}
      {activeTab === 4 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">{(Icons.Database as any)({ style: { color: '#8B5CF6' }, size: 24 })}</div>
            <h3>Veri Yedekleme</h3>
            <label className="settings-toggle-premium ml-auto">
              <input
                type="checkbox"
                checked={rules.dataBackup.enabled}
                onChange={(e) => updateRules({ dataBackup: { ...rules.dataBackup, enabled: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="settings-card-body">
            <div className="settings-grid-2">
              <div className="settings-form-group">
                <label>Yedekleme Sıklığı</label>
                <select
                  className="settings-input-premium"
                  value={rules.dataBackup.frequency}
                  onChange={(e) => updateRules({ dataBackup: { ...rules.dataBackup, frequency: e.target.value } })}
                >
                  <option value="hourly">Saatlik</option>
                  <option value="daily">Günlük</option>
                  <option value="weekly">Haftalık</option>
                </select>
              </div>

              <div className="settings-form-group">
                <label>Saklama Süresi (gün)</label>
                <input
                  type="number"
                  className="settings-input-premium"
                  value={rules.dataBackup.keepDays}
                  onChange={(e) => updateRules({ dataBackup: { ...rules.dataBackup, keepDays: Number(e.target.value) } })}
                  min="1"
                  max="365"
                />
              </div>
            </div>

            <label className="settings-toggle-premium">
              <input
                type="checkbox"
                checked={rules.dataBackup.autoCleanup}
                onChange={(e) => updateRules({ dataBackup: { ...rules.dataBackup, autoCleanup: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
              <span className="toggle-label">Otomatik temizlik</span>
            </label>

            {rules.dataBackup.lastBackup && (
              <div className="settings-alert-success">
                <div className="text-sm text-gray-400">Son Yedekleme</div>
                <div className="text-green-400 font-semibold">{new Date(rules.dataBackup.lastBackup).toLocaleString('tr-TR')}</div>
              </div>
            )}

            <button
              onClick={triggerManualBackup}
              className="settings-btn-primary w-full"
            >
              {(Icons.Download as any)({ size: 20 })}
              <span>Manuel Yedekleme Başlat</span>
            </button>
          </div>
        </div>
      )}

      {/* Tab 5: Webhooks */}
      {activeTab === 5 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">{(Icons.Webhook as any)({ style: { color: '#F59E0B' }, size: 24 })}</div>
            <h3>Webhook Entegrasyonları</h3>
            <label className="settings-toggle-premium ml-auto">
              <input
                type="checkbox"
                checked={rules.webhooks.enabled}
                onChange={(e) => updateRules({ webhooks: { ...rules.webhooks, enabled: e.target.checked } })}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="settings-card-body">
            {rules.webhooks.endpoints.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                {(Icons.Webhook as any)({ size: 48, className: "mx-auto mb-4 opacity-50" })}
                <p>Henüz webhook eklenmemiş</p>
              </div>
            ) : (
              <div className="space-y-4">
                {rules.webhooks.endpoints.map((webhook: any) => (
                  <div key={webhook.id} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <div className="font-semibold text-lg">{webhook.name}</div>
                        <div className="text-sm text-gray-400 mt-1">{webhook.url}</div>
                      </div>
                      <label className="settings-toggle-premium">
                        <input type="checkbox" checked={webhook.active} readOnly />
                        <span className="toggle-slider"></span>
                      </label>
                    </div>
                    <div className="flex gap-2 flex-wrap">
                      {webhook.events.map((event: string) => (
                        <span key={event} className="px-3 py-1 bg-black text-white border border-white/50 rounded-full text-xs">
                          {event}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Save Status */}
      {saveStatus === 'saving' && (
        <div className="settings-alert-info">
          {(Icons.Loader2 as any)({ className: "animate-spin", size: 16 })}
          <span>Kaydediliyor...</span>
        </div>
      )}
      {saveStatus === 'success' && (
        <div className="settings-alert-success">
          {(Icons.Check as any)({ size: 16 })}
          <span>✅ Kaydedildi!</span>
        </div>
      )}
    </div>
  );
}
