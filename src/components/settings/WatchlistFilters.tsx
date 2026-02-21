'use client';

import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';

interface WatchlistFiltersProps {
  onSave?: () => void;
}

export default function WatchlistFilters({ onSave }: WatchlistFiltersProps) {
  const translations = {
    failedToLoad: 'Filtreler yüklenemedi',
    watchlistAndBlacklist: 'İzleme Listesi ve Kara Liste',
    priceAndVolume: 'Fiyat ve Hacim',
    signalsAndVolatility: 'Sinyaller ve Volatilite',
    quickFilters: 'Hızlı Filtreler',
    watchlist: 'İzleme Listesi',
    enable: 'Etkinleştir',
    addCoin: 'Coin Ekle',
    noCoinInWatchlist: 'İzleme listesinde coin yok',
    onlyShowWatchlist: 'Sadece izleme listesindeki coinleri göster (diğerlerini gizle)',
    blacklist: 'Kara Liste',
    addCoinToIgnore: 'Yok Sayılacak Coin Ekle',
    coinToIgnore: 'Yok sayılacak coin...',
    noCoinsBlacklisted: 'Kara listede coin yok',
    priceFilter: 'Fiyat Filtresi',
    minPrice: 'Min Fiyat',
    maxPrice: 'Max Fiyat',
    volumeFilter: 'Hacim Filtresi',
    min24hVolume: 'Min 24s Hacim (USDT)',
    signalFilter: 'Sinyal Filtresi',
    minConfidence: 'Min Güven',
    volatilityFilter: 'Volatilite Filtresi',
    minVolatility: 'Min Volatilite (%)',
    maxVolatility: 'Max Volatilite (%)',
    gainersOnly: 'Sadece Yükselenler',
    losersOnly: 'Sadece Düşenler',
    highVolume: 'Yüksek Hacim',
    topMovers: 'En Çok Hareket Edenler',
    saving: 'Kaydediliyor...',
  };

  const [filters, setFilters] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [activeTab, setActiveTab] = useState(1);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [newCoin, setNewCoin] = useState('');

  useEffect(() => {
    fetchFilters();
  }, []);

  const fetchFilters = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/watchlist-filters');
      const data = await response.json();
      if (data.success) {
        setFilters(data.data.filters);
      }
    } catch (error) {
      showMessage('error', translations.failedToLoad);
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  useEffect(() => {
    if (!loading && filters) {
      const timer = setTimeout(() => saveFilters(), 800);
      return () => clearTimeout(timer);
    }
  }, [filters]);

  const saveFilters = async () => {
    if (!filters) return;
    try {
      setSaving(true);
      await fetch('/api/watchlist-filters', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(filters),
      });
      onSave?.();
    } finally {
      setSaving(false);
    }
  };

  const addToWatchlist = async () => {
    if (!newCoin.trim()) return;
    const coin = newCoin.toUpperCase().trim();
    const response = await fetch('/api/watchlist-filters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add_to_watchlist', coin }),
    });
    const data = await response.json();
    if (data.success) {
      setFilters(data.data);
      setNewCoin('');
      showMessage('success', data.message);
    }
  };

  const removeFromWatchlist = async (coin: string) => {
    const response = await fetch('/api/watchlist-filters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'remove_from_watchlist', coin }),
    });
    const data = await response.json();
    if (data.success) setFilters(data.data);
  };

  const addToBlacklist = async () => {
    if (!newCoin.trim()) return;
    const coin = newCoin.toUpperCase().trim();
    const response = await fetch('/api/watchlist-filters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'add_to_blacklist', coin }),
    });
    const data = await response.json();
    if (data.success) {
      setFilters(data.data);
      setNewCoin('');
      showMessage('success', data.message);
    }
  };

  const removeFromBlacklist = async (coin: string) => {
    const response = await fetch('/api/watchlist-filters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'remove_from_blacklist', coin }),
    });
    const data = await response.json();
    if (data.success) setFilters(data.data);
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

  if (!filters) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">{translations.failedToLoad}</div>
      </div>
    );
  }

  return (
    <div className="settings-content-wrapper">
      {message && (
        <div className={message.type === 'success' ? 'settings-alert-success' : 'settings-alert-error'}>
          {message.text}
        </div>
      )}

      <div className="flex gap-2 mb-6 overflow-x-auto">
        {[
          { id: 1, label: translations.watchlistAndBlacklist, icon: Icons.Star },
          { id: 2, label: translations.priceAndVolume, icon: Icons.DollarSign },
          { id: 3, label: translations.signalsAndVolatility, icon: Icons.TrendingUp },
          { id: 4, label: translations.quickFilters, icon: Icons.Zap },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${
              activeTab === tab.id
                ? 'bg-black text-white border-2 border-white'
                : 'bg-black text-gray-400 hover:bg-black border border-white/30'
            }`}
          >
            {(tab.icon as any)({ size: 16 })}
            <span className="whitespace-nowrap">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* TAB 1: WATCHLIST & BLACKLIST */}
      {activeTab === 1 && (
        <div className="space-y-6">
          <div className="settings-premium-card">
            <div className="settings-card-header">
              <div className="settings-card-icon">
                {(Icons.Star as any)({ style: { color: '#F59E0B' }, size: 24 })}
              </div>
              <h3>{translations.watchlist} ({filters.watchlist.coins.length})</h3>
              <label className="settings-toggle-premium ml-auto">
                <input
                  type="checkbox"
                  checked={filters.watchlist.enabled}
                  onChange={(e) => setFilters({ ...filters, watchlist: { ...filters.watchlist, enabled: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.enable}</span>
              </label>
            </div>

            <div className="settings-card-body">
              <div className="settings-form-group">
                <label>{translations.addCoin}</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newCoin}
                    onChange={(e) => setNewCoin(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addToWatchlist()}
                    placeholder="BTCUSDT, ETHUSDT..."
                    className="settings-input-premium flex-1"
                  />
                  <button
                    onClick={addToWatchlist}
                    className="settings-btn-primary"
                  >
                    {(Icons.Plus as any)({ size: 18 })}
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 mb-4">
                {filters.watchlist.coins.map((coin: string) => (
                  <div key={coin} className="bg-black text-white border border-white/50 px-3 py-1 rounded-lg flex items-center gap-2">
                    <span>{coin}</span>
                    <button onClick={() => removeFromWatchlist(coin)} className="hover:text-gray-400">
                      {(Icons.X as any)({ size: 14 })}
                    </button>
                  </div>
                ))}
                {filters.watchlist.coins.length === 0 && (
                  <p className="text-gray-500 text-sm">{translations.noCoinInWatchlist}</p>
                )}
              </div>

              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={filters.watchlist.onlyShowWatchlist}
                  onChange={(e) =>
                    setFilters({ ...filters, watchlist: { ...filters.watchlist, onlyShowWatchlist: e.target.checked } })
                  }
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.onlyShowWatchlist}</span>
              </label>
            </div>
          </div>

          <div className="settings-premium-card">
            <div className="settings-card-header">
              <div className="settings-card-icon">
                {(Icons.Ban as any)({ style: { color: '#EF4444' }, size: 24 })}
              </div>
              <h3>{translations.blacklist} ({filters.blacklist.coins.length})</h3>
              <label className="settings-toggle-premium ml-auto">
                <input
                  type="checkbox"
                  checked={filters.blacklist.enabled}
                  onChange={(e) => setFilters({ ...filters, blacklist: { ...filters.blacklist, enabled: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.enable}</span>
              </label>
            </div>

            <div className="settings-card-body">
              <div className="settings-form-group">
                <label>{translations.addCoinToIgnore}</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={newCoin}
                    onChange={(e) => setNewCoin(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && addToBlacklist()}
                    placeholder={translations.coinToIgnore}
                    className="settings-input-premium flex-1"
                  />
                  <button onClick={addToBlacklist} className="settings-btn-secondary">
                    {(Icons.Plus as any)({ size: 18 })}
                  </button>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {filters.blacklist.coins.map((coin: string) => (
                  <div key={coin} className="bg-black text-white border border-white/50 px-3 py-1 rounded-lg flex items-center gap-2">
                    <span>{coin}</span>
                    <button onClick={() => removeFromBlacklist(coin)} className="hover:text-gray-400">
                      {(Icons.X as any)({ size: 14 })}
                    </button>
                  </div>
                ))}
                {filters.blacklist.coins.length === 0 && <p className="text-gray-500 text-sm">{translations.noCoinsBlacklisted}</p>}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: PRICE & VOLUME */}
      {activeTab === 2 && (
        <div className="space-y-6">
          <div className="settings-premium-card">
            <div className="settings-card-header">
              <div className="settings-card-icon">
                {(Icons.DollarSign as any)({ style: { color: '#10B981' }, size: 24 })}
              </div>
              <h3>{translations.priceFilter}</h3>
              <label className="settings-toggle-premium ml-auto">
                <input
                  type="checkbox"
                  checked={filters.priceFilter.enabled}
                  onChange={(e) => setFilters({ ...filters, priceFilter: { ...filters.priceFilter, enabled: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.enable}</span>
              </label>
            </div>
            <div className="settings-card-body">
              <div className="settings-grid-2">
                <div className="settings-form-group">
                  <label>{translations.minPrice}</label>
                  <input
                    type="number"
                    value={filters.priceFilter.minPrice}
                    onChange={(e) => setFilters({ ...filters, priceFilter: { ...filters.priceFilter, minPrice: Number(e.target.value) } })}
                    className="settings-input-premium"
                  />
                </div>
                <div className="settings-form-group">
                  <label>{translations.maxPrice}</label>
                  <input
                    type="number"
                    value={filters.priceFilter.maxPrice}
                    onChange={(e) => setFilters({ ...filters, priceFilter: { ...filters.priceFilter, maxPrice: Number(e.target.value) } })}
                    className="settings-input-premium"
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="settings-premium-card">
            <div className="settings-card-header">
              <div className="settings-card-icon">
                {(Icons.BarChart3 as any)({ style: { color: '#3B82F6' }, size: 24 })}
              </div>
              <h3>{translations.volumeFilter}</h3>
              <label className="settings-toggle-premium ml-auto">
                <input
                  type="checkbox"
                  checked={filters.volumeFilter.enabled}
                  onChange={(e) => setFilters({ ...filters, volumeFilter: { ...filters.volumeFilter, enabled: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.enable}</span>
              </label>
            </div>
            <div className="settings-card-body">
              <div className="settings-form-group">
                <label>{translations.min24hVolume}</label>
                <input
                  type="number"
                  value={filters.volumeFilter.min24hVolume}
                  onChange={(e) => setFilters({ ...filters, volumeFilter: { ...filters.volumeFilter, min24hVolume: Number(e.target.value) } })}
                  className="settings-input-premium"
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: SIGNALS & VOLATILITY */}
      {activeTab === 3 && (
        <div className="space-y-6">
          <div className="settings-premium-card">
            <div className="settings-card-header">
              <div className="settings-card-icon">
                {(Icons.Target as any)({ style: { color: '#EF4444' }, size: 24 })}
              </div>
              <h3>{translations.signalFilter}</h3>
              <label className="settings-toggle-premium ml-auto">
                <input
                  type="checkbox"
                  checked={filters.signalFilter.enabled}
                  onChange={(e) => setFilters({ ...filters, signalFilter: { ...filters.signalFilter, enabled: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.enable}</span>
              </label>
            </div>
            <div className="settings-card-body">
              <div className="settings-form-group">
                <label>{translations.minConfidence}: {filters.signalFilter.minConfidence}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={filters.signalFilter.minConfidence}
                  onChange={(e) => setFilters({ ...filters, signalFilter: { ...filters.signalFilter, minConfidence: Number(e.target.value) } })}
                  className="settings-slider-premium"
                />
              </div>
            </div>
          </div>

          <div className="settings-premium-card">
            <div className="settings-card-header">
              <div className="settings-card-icon">
                {(Icons.TrendingUp as any)({ style: { color: '#10B981' }, size: 24 })}
              </div>
              <h3>{translations.volatilityFilter}</h3>
              <label className="settings-toggle-premium ml-auto">
                <input
                  type="checkbox"
                  checked={filters.volatilityFilter.enabled}
                  onChange={(e) => setFilters({ ...filters, volatilityFilter: { ...filters.volatilityFilter, enabled: e.target.checked } })}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">{translations.enable}</span>
              </label>
            </div>
            <div className="settings-card-body">
              <div className="settings-grid-2">
                <div className="settings-form-group">
                  <label>{translations.minVolatility}</label>
                  <input
                    type="number"
                    value={filters.volatilityFilter.minVolatility}
                    onChange={(e) => setFilters({ ...filters, volatilityFilter: { ...filters.volatilityFilter, minVolatility: Number(e.target.value) } })}
                    className="settings-input-premium"
                  />
                </div>
                <div className="settings-form-group">
                  <label>{translations.maxVolatility}</label>
                  <input
                    type="number"
                    value={filters.volatilityFilter.maxVolatility}
                    onChange={(e) => setFilters({ ...filters, volatilityFilter: { ...filters.volatilityFilter, maxVolatility: Number(e.target.value) } })}
                    className="settings-input-premium"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: QUICK FILTERS */}
      {activeTab === 4 && (
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">
              {(Icons.Zap as any)({ style: { color: '#F59E0B' }, size: 24 })}
            </div>
            <h3>{translations.quickFilters}</h3>
          </div>
          <div className="settings-card-body">
            <div className="settings-grid-2">
              {[
                { key: 'showGainersOnly', label: translations.gainersOnly, icon: Icons.TrendingUp },
                { key: 'showLosersOnly', label: translations.losersOnly, icon: Icons.TrendingDown },
                { key: 'showHighVolumeOnly', label: translations.highVolume, icon: Icons.BarChart3 },
                { key: 'showTopMovers', label: translations.topMovers, icon: Icons.Zap },
              ].map((filter) => (
                <label
                  key={filter.key}
                  className="settings-toggle-premium"
                >
                  <input
                    type="checkbox"
                    checked={filters.quickFilters[filter.key]}
                    onChange={(e) => setFilters({ ...filters, quickFilters: { ...filters.quickFilters, [filter.key]: e.target.checked } })}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label flex items-center gap-2">
                    {(filter.icon as any)({ size: 16 })}
                    {filter.label}
                  </span>
                </label>
              ))}
            </div>
          </div>
        </div>
      )}

      {saving && (
        <div className="fixed bottom-4 right-4 bg-black border-2 border-white rounded-lg px-4 py-2 flex items-center gap-2 shadow-lg">
          {(Icons.Loader2 as any)({ size: 16, className: "animate-spin text-white" })}
          <span className="text-sm text-white">{translations.saving}</span>
        </div>
      )}
    </div>
  );
}
