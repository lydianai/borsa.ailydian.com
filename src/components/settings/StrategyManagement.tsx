'use client';

import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';

interface BacktestStats {
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  totalTrades: number;
  avgProfit: number;
}

interface StrategyHealth {
  status: 'healthy' | 'degraded' | 'offline';
  lastCheck: string;
  responseTime: number;
  errorRate: number;
}

interface Strategy {
  id: string;
  name: string;
  description: string;
  category: 'AI' | 'Technical' | 'Market' | 'Advanced';
  endpoint: string;
  enabled: boolean;
  weight: number;
  minConfidence: number;
  backtestStats?: BacktestStats;
  health: StrategyHealth;
}

interface Summary {
  totalStrategies: number;
  enabledStrategies: number;
  categories: {
    AI: number;
    Technical: number;
    Market: number;
    Advanced: number;
  };
  avgWeight: number;
  health: {
    healthy: number;
    degraded: number;
    offline: number;
  };
}

interface StrategyManagementProps {
  onSave?: () => void;
}

type CategoryFilter = 'all' | 'AI' | 'Technical' | 'Market' | 'Advanced';

export default function StrategyManagement({ onSave }: StrategyManagementProps) {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [categoryFilter, setCategoryFilter] = useState<CategoryFilter>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [showHealthCheck, setShowHealthCheck] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Fetch strategies
  useEffect(() => {
    fetchStrategies();
  }, []);

  const fetchStrategies = async (checkHealth = false) => {
    try {
      setLoading(true);
      const url = `/api/strategy-management${checkHealth ? '?checkHealth=true' : ''}`;
      const response = await fetch(url);
      const data = await response.json();

      if (data.success) {
        setStrategies(data.data.strategies);
        setSummary(data.data.summary);
      } else {
        showMessage('error', data.error || 'Stratejiler yüklenemedi');
      }
    } catch (error) {
      showMessage('error', 'Stratejiler alınamadı');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  // Auto-save on change (debounced)
  useEffect(() => {
    if (!loading && strategies.length > 0) {
      const timer = setTimeout(() => {
        saveStrategies();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [strategies]);

  const saveStrategies = async () => {
    try {
      setSaving(true);
      const response = await fetch('/api/strategy-management', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bulk: true,
          strategies: strategies.map((s) => ({
            id: s.id,
            enabled: s.enabled,
            weight: s.weight,
            minConfidence: s.minConfidence,
          })),
        }),
      });

      const data = await response.json();
      if (data.success) {
        onSave?.();
      }
    } catch (error) {
      console.error('Save error:', error);
    } finally {
      setSaving(false);
    }
  };

  const updateStrategy = (id: string, updates: Partial<Strategy>) => {
    setStrategies((prev) =>
      prev.map((s) => (s.id === id ? { ...s, ...updates } : s))
    );
  };

  const bulkEnableByCategory = (category: CategoryFilter, enabled: boolean) => {
    setStrategies((prev) =>
      prev.map((s) =>
        category === 'all' || s.category === category ? { ...s, enabled } : s
      )
    );
    showMessage('success', `${enabled ? 'Aktifleştirildi' : 'Devre dışı bırakıldı'} ${category === 'all' ? 'tüm' : category === 'Technical' ? 'teknik' : category === 'Market' ? 'piyasa' : category === 'Advanced' ? 'gelişmiş' : category} stratejiler`);
  };

  const resetToDefaults = async () => {
    if (!confirm('Tüm stratejiler varsayılan ayarlara döndürülecek. Devam edilsin mi?')) {
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/strategy-management', {
        method: 'PUT',
      });

      const data = await response.json();
      if (data.success) {
        await fetchStrategies();
        showMessage('success', 'Stratejiler varsayılana sıfırlandı');
      } else {
        showMessage('error', data.error);
      }
    } catch (error) {
      showMessage('error', 'Stratejiler sıfırlanamadı');
    } finally {
      setLoading(false);
    }
  };

  const runHealthCheck = async () => {
    setShowHealthCheck(true);
    await fetchStrategies(true);
    setShowHealthCheck(false);
    showMessage('success', 'Sağlık kontrolü tamamlandı');
  };

  // Filter strategies
  const filteredStrategies = strategies.filter((s) => {
    const matchesCategory = categoryFilter === 'all' || s.category === categoryFilter;
    const matchesSearch =
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  // Health status icon
  const getHealthIcon = (status: StrategyHealth['status']) => {
    switch (status) {
      case 'healthy':
        return (Icons.CheckCircle as any)({ size: 16, className: "text-green-500" });
      case 'degraded':
        return (Icons.AlertCircle as any)({ size: 16, className: "text-yellow-500" });
      case 'offline':
        return (Icons.XCircle as any)({ size: 16, className: "text-red-500" });
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

  return (
    <div className="settings-content-wrapper">
      {/* Message */}
      {message && (
        <div className={message.type === 'success' ? 'settings-alert-success' : 'settings-alert-error'}>
          {message.text}
        </div>
      )}

      {/* Summary Dashboard */}
      {summary && (
        <div className="settings-grid-3 mb-6">
          <div className="settings-stat-card">
            <div className="stat-icon">{(Icons.BarChart3 as any)({ style: { color: '#3B82F6' }, size: 32 })}</div>
            <div className="stat-value">{summary.totalStrategies}</div>
            <div className="stat-label">Toplam Strateji</div>
          </div>
          <div className="settings-stat-card">
            <div className="stat-icon">{(Icons.CheckCircle2 as any)({ style: { color: '#10B981' }, size: 32 })}</div>
            <div className="stat-value">{summary.enabledStrategies}</div>
            <div className="stat-label">Aktif</div>
          </div>
          <div className="settings-stat-card">
            <div className="stat-icon">{(Icons.Scale as any)({ style: { color: '#F59E0B' }, size: 32 })}</div>
            <div className="stat-value">{summary.avgWeight}%</div>
            <div className="stat-label">Ort. Ağırlık</div>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="settings-premium-card mb-6">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.Search as any)({ style: { color: '#6B7280' }, size: 20 })}</div>
          <h3>Filtreler ve İşlemler</h3>
        </div>
        <div className="settings-card-body">
          {/* Category Filters */}
          <div className="flex flex-wrap gap-2 mb-4">
            {(['all', 'AI', 'Technical', 'Market', 'Advanced'] as CategoryFilter[]).map((cat) => (
              <button
                key={cat}
                onClick={() => setCategoryFilter(cat)}
                className={`px-4 py-2 rounded-lg transition-all ${
                  categoryFilter === cat
                    ? 'bg-black text-white border-2 border-white'
                    : 'bg-black text-gray-400 hover:bg-black border border-white/30'
                }`}
              >
                {cat === 'all' ? 'Tümü' : cat === 'Technical' ? 'Teknik' : cat === 'Market' ? 'Piyasa' : cat === 'Advanced' ? 'Gelişmiş' : cat}
                {summary && cat !== 'all' && (
                  <span className="ml-2 text-xs opacity-60">({summary.categories[cat]})</span>
                )}
              </button>
            ))}
          </div>

          {/* Search */}
          <div className="settings-form-group mb-4">
            <label>Strateji Ara</label>
            <div className="relative">
              {(Icons.Search as any)({ className: "absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500", size: 18 })}
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Strateji ara..."
                className="settings-input-premium pl-10"
              />
            </div>
          </div>

          {/* Bulk Actions */}
          <div className="flex gap-2">
            <button
              onClick={() => bulkEnableByCategory(categoryFilter, true)}
              className="settings-btn-primary flex-1"
            >
              {(Icons.CheckCircle as any)({ size: 16 })}
              <span>Aktifleştir {categoryFilter === 'all' ? 'Tümü' : categoryFilter === 'Technical' ? 'Teknik' : categoryFilter === 'Market' ? 'Piyasa' : categoryFilter === 'Advanced' ? 'Gelişmiş' : categoryFilter}</span>
            </button>
            <button
              onClick={() => bulkEnableByCategory(categoryFilter, false)}
              className="settings-btn-secondary flex-1"
            >
              {(Icons.XCircle as any)({ size: 16 })}
              <span>Devre Dışı Bırak {categoryFilter === 'all' ? 'Tümü' : categoryFilter === 'Technical' ? 'Teknik' : categoryFilter === 'Market' ? 'Piyasa' : categoryFilter === 'Advanced' ? 'Gelişmiş' : categoryFilter}</span>
            </button>
          </div>

          <div className="flex gap-2 mt-4">
            <button
              onClick={runHealthCheck}
              disabled={showHealthCheck}
              className="settings-btn-secondary flex-1"
            >
              {showHealthCheck ? (Icons.Loader2 as any)({ size: 16, className: "animate-spin" }) : (Icons.Activity as any)({ size: 16 })}
              <span>Sağlık Kontrolü</span>
            </button>
            <button
              onClick={resetToDefaults}
              className="settings-btn-secondary flex-1"
            >
              {(Icons.RotateCcw as any)({ size: 16 })}
              <span>Sıfırla</span>
            </button>
          </div>
        </div>
      </div>

      {/* Strategy Grid */}
      <div className="space-y-4">
        {filteredStrategies.map((strategy) => (
          <div
            key={strategy.id}
            className={`settings-premium-card ${
              strategy.enabled ? 'border-cyan-600/50' : 'opacity-60'
            }`}
          >
            {/* Header */}
            <div className="settings-card-header">
              <div className="flex items-center gap-3 flex-1">
                <div className="settings-card-icon">{
                  strategy.category === 'AI' ? (Icons.Bot as any)({ style: { color: '#8B5CF6' }, size: 16 }) :
                  strategy.category === 'Technical' ? (Icons.BarChart3 as any)({ style: { color: '#3B82F6' }, size: 16 }) :
                  strategy.category === 'Market' ? (Icons.TrendingUp as any)({ style: { color: '#10B981' }, size: 16 }) :
                  (Icons.Zap as any)({ style: { color: '#F59E0B' }, size: 16 })
                }</div>
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <h3>{strategy.name}</h3>
                    <span className="px-2 py-1 rounded text-xs font-semibold bg-black text-white border border-white/50">
                      {strategy.category === 'AI' ? 'AI' : strategy.category === 'Technical' ? 'Teknik' : strategy.category === 'Market' ? 'Piyasa' : 'Gelişmiş'}
                    </span>
                    {getHealthIcon(strategy.health.status)}
                  </div>
                  <p className="text-gray-400 text-sm">{strategy.description}</p>
                </div>
              </div>

              {/* Enable Toggle */}
              <label className="settings-toggle-premium ml-4">
                <input
                  type="checkbox"
                  checked={strategy.enabled}
                  onChange={(e) => updateStrategy(strategy.id, { enabled: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>

            <div className="settings-card-body">
              {/* Controls */}
              <div className="settings-grid-2 mb-4">
                {/* Weight Slider */}
                <div className="settings-form-group">
                  <label>Ağırlık: {strategy.weight}%</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    step="5"
                    value={strategy.weight}
                    onChange={(e) => updateStrategy(strategy.id, { weight: Number(e.target.value) })}
                    disabled={!strategy.enabled}
                    className="settings-slider-premium"
                  />
                </div>

                {/* Min Confidence Slider */}
                <div className="settings-form-group">
                  <label>Min. Güven: {strategy.minConfidence}%</label>
                  <input
                    type="range"
                    min="50"
                    max="100"
                    step="5"
                    value={strategy.minConfidence}
                    onChange={(e) => updateStrategy(strategy.id, { minConfidence: Number(e.target.value) })}
                    disabled={!strategy.enabled}
                    className="settings-slider-premium"
                  />
                </div>
              </div>

              {/* Backtest Stats */}
              {strategy.backtestStats && (
                <div className="bg-black border border-white/30 rounded-lg p-4 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Kazanma Oranı</div>
                    <div className="text-sm font-bold text-white">{strategy.backtestStats.winRate}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Kar Faktörü</div>
                    <div className="text-sm font-bold text-white">{strategy.backtestStats.profitFactor.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Sharpe Oranı</div>
                    <div className="text-sm font-bold text-white">{strategy.backtestStats.sharpeRatio.toFixed(1)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Maks. Düşüş</div>
                    <div className="text-sm font-bold text-white">{strategy.backtestStats.maxDrawdown.toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Toplam İşlem</div>
                    <div className="text-sm font-bold text-white">{strategy.backtestStats.totalTrades}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400 mb-1">Ort. Kar</div>
                    <div className="text-sm font-bold text-white">{strategy.backtestStats.avgProfit.toFixed(1)}%</div>
                  </div>
                </div>
              )}

              {/* Health Info */}
              <div className="mt-4 flex items-center gap-4 text-xs text-gray-500">
                <span>Yanıt Süresi: {strategy.health.responseTime}ms</span>
                <span>Son Kontrol: {new Date(strategy.health.lastCheck).toLocaleTimeString()}</span>
                <span>Hata Oranı: {strategy.health.errorRate}%</span>
              </div>
            </div>
          </div>
        ))}

        {filteredStrategies.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            {(Icons.Search as any)({ size: 48, className: "mx-auto mb-4 opacity-50" })}
            <p>Filtrelerinize uygun strateji bulunamadı</p>
          </div>
        )}
      </div>

      {/* Auto-save indicator */}
      {saving && (
        <div className="fixed bottom-4 right-4 bg-black border-2 border-white rounded-lg px-4 py-2 flex items-center gap-2 shadow-lg">
          {(Icons.Loader2 as any)({ size: 16, className: "animate-spin text-white" })}
          <span className="text-sm text-white">Kaydediliyor...</span>
        </div>
      )}
    </div>
  );
}
