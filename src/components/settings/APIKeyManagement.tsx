'use client';

// @ts-nocheck - Temporary fix for lucide-react React 18 types compatibility
import React, { useState, useEffect } from 'react';
import * as Icons from 'lucide-react';

interface APIKeyManagementProps {
  onSave?: () => void;
}

interface BinanceConfig {
  apiKey: string;
  secretKey: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
  testnet: boolean;
}

interface OKXConfig {
  apiKey: string;
  secretKey: string;
  passphrase: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
  testnet: boolean;
}

interface BybitConfig {
  apiKey: string;
  secretKey: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
  testnet: boolean;
}

interface BTCTurkConfig {
  apiKey: string;
  secretKey: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
}

interface GroqConfig {
  apiKey: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
  model: 'llama-3.3-70b-versatile' | 'llama-3.1-8b-instant' | 'mixtral-8x7b-32768';
}

interface CoinMarketCapConfig {
  apiKey: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
  plan: 'free' | 'basic' | 'pro';
}

interface TelegramConfig {
  botToken: string;
  chatId: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
}

interface RapidAPIConfig {
  apiKey: string;
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'untested';
  lastTested: string | null;
}

interface APIKeys {
  binance: BinanceConfig;
  okx: OKXConfig;
  bybit: BybitConfig;
  btcturk: BTCTurkConfig;
  groq: GroqConfig;
  coinmarketcap: CoinMarketCapConfig;
  telegram: TelegramConfig;
  rapidapi: RapidAPIConfig;
}

export default function APIKeyManagement({ onSave }: APIKeyManagementProps) {
  // Check if user is admin (from localStorage for demo)
  const isAdmin = typeof window !== 'undefined' && localStorage.getItem('user-role') === 'admin';

  const [keys, setKeys] = useState<APIKeys | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [showPasswords, setShowPasswords] = useState<{ [key: string]: boolean }>({});

  // Fetch API keys
  useEffect(() => {
    fetchAPIKeys();
  }, []);

  const fetchAPIKeys = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/api-keys');
      const data = await response.json();

      if (data.success) {
        setKeys(data.data);
      } else {
        showMessage('error', data.error || 'API anahtarları yüklenemedi');
      }
    } catch (error) {
      showMessage('error', 'API anahtarları alınamadı');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 5000);
  };

  // Auto-save on change (debounced)
  useEffect(() => {
    if (!loading && keys) {
      const timer = setTimeout(() => {
        saveAPIKeys();
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [keys]);

  const saveAPIKeys = async () => {
    if (!keys) return;

    try {
      setSaving(true);
      const response = await fetch('/api/api-keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(keys),
      });

      const data = await response.json();
      if (data.success) {
        onSave?.();
      } else {
        if (data.details && Array.isArray(data.details)) {
          showMessage('error', data.details.join(', '));
        } else {
          showMessage('error', data.error || 'Kaydedilemedi');
        }
      }
    } catch (error) {
      console.error('Save error:', error);
    } finally {
      setSaving(false);
    }
  };

  const testConnection = async (service: string) => {
    if (!keys) return;

    setTesting(service);

    try {
      let testData: any = { action: 'test', service };

      if (service === 'binance') {
        testData = {
          ...testData,
          apiKey: keys.binance.apiKey,
          secretKey: keys.binance.secretKey,
          testnet: keys.binance.testnet,
        };
      } else if (service === 'okx') {
        testData = {
          ...testData,
          apiKey: keys.okx.apiKey,
          secretKey: keys.okx.secretKey,
          passphrase: keys.okx.passphrase,
          testnet: keys.okx.testnet,
        };
      } else if (service === 'bybit') {
        testData = {
          ...testData,
          apiKey: keys.bybit.apiKey,
          secretKey: keys.bybit.secretKey,
          testnet: keys.bybit.testnet,
        };
      } else if (service === 'btcturk') {
        testData = {
          ...testData,
          apiKey: keys.btcturk.apiKey,
          secretKey: keys.btcturk.secretKey,
        };
      } else if (service === 'groq') {
        testData = { ...testData, apiKey: keys.groq.apiKey };
      } else if (service === 'coinmarketcap') {
        testData = { ...testData, apiKey: keys.coinmarketcap.apiKey };
      } else if (service === 'telegram') {
        testData = {
          ...testData,
          botToken: keys.telegram.botToken,
          chatId: keys.telegram.chatId,
        };
      } else if (service === 'rapidapi') {
        testData = { ...testData, apiKey: keys.rapidapi.apiKey };
      }

      const response = await fetch('/api/api-keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(testData),
      });

      const result = await response.json();

      if (result.success) {
        showMessage('success', result.message);
        // Reload to get updated status
        await fetchAPIKeys();
      } else {
        showMessage('error', result.message);
      }
    } catch (error) {
      showMessage('error', 'Bağlantı testi başarısız');
      console.error(error);
    } finally {
      setTesting(null);
    }
  };

  const resetToDefaults = async () => {
    if (!confirm('Tüm API anahtarları silinecek. Devam etmek istiyor musunuz?')) {
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/api-keys', {
        method: 'PUT',
      });

      const data = await response.json();
      if (data.success) {
        await fetchAPIKeys();
        showMessage('success', 'API anahtarları varsayılanlara sıfırlandı');
      } else {
        showMessage('error', data.error);
      }
    } catch (error) {
      showMessage('error', 'API anahtarları sıfırlanamadı');
    } finally {
      setLoading(false);
    }
  };

  const togglePasswordVisibility = (key: string) => {
    setShowPasswords((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const getStatusIcon = (status: 'connected' | 'disconnected' | 'untested') => {
    switch (status) {
      case 'connected':
        return <Icons.CheckCircle className="w-4 h-4 text-green-500" />;
      case 'disconnected':
        return <Icons.XCircle className="w-4 h-4 text-red-500" />;
      case 'untested':
        return <Icons.HelpCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const updateBinance = (updates: Partial<BinanceConfig>) => {
    if (!keys) return;
    // Auto-enable when API/secret key is filled
    if ((updates.apiKey || updates.secretKey) && (keys.binance.apiKey || updates.apiKey) && (keys.binance.secretKey || updates.secretKey)) {
      updates.enabled = true;
    }
    setKeys({ ...keys, binance: { ...keys.binance, ...updates } });
  };

  const updateOKX = (updates: Partial<OKXConfig>) => {
    if (!keys) return;
    // Auto-enable when API/secret/passphrase are filled
    if ((updates.apiKey || updates.secretKey || updates.passphrase) && (keys.okx.apiKey || updates.apiKey) && (keys.okx.secretKey || updates.secretKey) && (keys.okx.passphrase || updates.passphrase)) {
      updates.enabled = true;
    }
    setKeys({ ...keys, okx: { ...keys.okx, ...updates } });
  };

  const updateBybit = (updates: Partial<BybitConfig>) => {
    if (!keys) return;
    // Auto-enable when API/secret key is filled
    if ((updates.apiKey || updates.secretKey) && (keys.bybit.apiKey || updates.apiKey) && (keys.bybit.secretKey || updates.secretKey)) {
      updates.enabled = true;
    }
    setKeys({ ...keys, bybit: { ...keys.bybit, ...updates } });
  };

  const updateBTCTurk = (updates: Partial<BTCTurkConfig>) => {
    if (!keys) return;
    // Auto-enable when API/secret key is filled
    if ((updates.apiKey || updates.secretKey) && (keys.btcturk.apiKey || updates.apiKey) && (keys.btcturk.secretKey || updates.secretKey)) {
      updates.enabled = true;
    }
    setKeys({ ...keys, btcturk: { ...keys.btcturk, ...updates } });
  };

  const updateGroq = (updates: Partial<GroqConfig>) => {
    if (!keys) return;
    setKeys({ ...keys, groq: { ...keys.groq, ...updates } });
  };

  const updateCoinMarketCap = (updates: Partial<CoinMarketCapConfig>) => {
    if (!keys) return;
    setKeys({ ...keys, coinmarketcap: { ...keys.coinmarketcap, ...updates } });
  };

  const updateTelegram = (updates: Partial<TelegramConfig>) => {
    if (!keys) return;
    setKeys({ ...keys, telegram: { ...keys.telegram, ...updates } });
  };

  const updateRapidAPI = (updates: Partial<RapidAPIConfig>) => {
    if (!keys) return;
    setKeys({ ...keys, rapidapi: { ...keys.rapidapi, ...updates } });
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

  if (!keys) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">
          API anahtarları yüklenemedi. Lütfen sayfayı yenileyin.
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

      {/* Reset Button */}
      <div className="flex justify-end mb-6">
        <button onClick={resetToDefaults} className="settings-btn-secondary" style={{
          padding: '8px 16px',
          background: 'linear-gradient(135deg, #EF4444, #DC2626)',
          color: '#FFFFFF',
          border: 'none',
          borderRadius: '6px',
          fontWeight: '600',
          fontSize: '13px',
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
        }}>
          <Icons.Trash2 className="w-4 h-4" />
          <span>Tümünü Sıfırla</span>
        </button>
      </div>

      {/* API Services */}
      <div className="space-y-6">
        {/* 1. BINANCE */}
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">
              <Icons.TrendingUp style={{ color: '#F3BA2F' }} className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3>Binance Futures API</h3>
              <p className="text-gray-400 text-sm">Trading ve market data</p>
            </div>
            <div className="flex items-center gap-3">
              {getStatusIcon(keys.binance.status)}
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={keys.binance.enabled}
                  onChange={(e) => updateBinance({ enabled: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>

          <div className="settings-card-body">
            <div className="settings-form-group">
              <label>API Key</label>
              <div className="relative">
                <input
                  type={showPasswords['binance-api'] ? 'text' : 'password'}
                  value={keys.binance.apiKey}
                  onChange={(e) => updateBinance({ apiKey: e.target.value })}
                  placeholder="Binance API Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('binance-api')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['binance-api'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="settings-form-group">
              <label>Secret Key</label>
              <div className="relative">
                <input
                  type={showPasswords['binance-secret'] ? 'text' : 'password'}
                  value={keys.binance.secretKey}
                  onChange={(e) => updateBinance({ secretKey: e.target.value })}
                  placeholder="Binance Gizli Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('binance-secret')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['binance-secret'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={keys.binance.testnet}
                  onChange={(e) => updateBinance({ testnet: e.target.checked })}
                  className="w-4 h-4 text-yellow-600 bg-gray-700 border-gray-600 rounded"
                />
                <span className="text-sm text-gray-300">Testnet Kullan</span>
              </label>

              <button
                onClick={() => testConnection('binance')}
                disabled={testing === 'binance' || !keys.binance.apiKey || !keys.binance.secretKey}
                style={{
                  padding: '8px 16px',
                  background: testing === 'binance' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: '600',
                  fontSize: '13px',
                  cursor: testing === 'binance' ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  opacity: (!keys.binance.apiKey || !keys.binance.secretKey) ? 0.5 : 1,
                }}
              >
                {testing === 'binance' ? (
                  <Icons.Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Icons.Zap className="w-4 h-4" />
                )}
                <span>Bağlantıyı Test Et</span>
              </button>
            </div>

            {keys.binance.lastTested && (
              <div className="text-xs text-gray-500">
                Son test: {new Date(keys.binance.lastTested).toLocaleString()}
              </div>
            )}
          </div>
        </div>

        {/* 2. OKX */}
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">
              <Icons.TrendingUp style={{ color: '#000000' }} className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3>OKX API</h3>
              <p className="text-gray-400 text-sm">Trading ve market data</p>
            </div>
            <div className="flex items-center gap-3">
              {getStatusIcon(keys.okx.status)}
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={keys.okx.enabled}
                  onChange={(e) => updateOKX({ enabled: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>

          <div className="settings-card-body">
            <div className="settings-form-group">
              <label>API Key</label>
              <div className="relative">
                <input
                  type={showPasswords['okx-api'] ? 'text' : 'password'}
                  value={keys.okx.apiKey}
                  onChange={(e) => updateOKX({ apiKey: e.target.value })}
                  placeholder="OKX API Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('okx-api')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['okx-api'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="settings-form-group">
              <label>Secret Key</label>
              <div className="relative">
                <input
                  type={showPasswords['okx-secret'] ? 'text' : 'password'}
                  value={keys.okx.secretKey}
                  onChange={(e) => updateOKX({ secretKey: e.target.value })}
                  placeholder="OKX Gizli Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('okx-secret')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['okx-secret'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="settings-form-group">
              <label>Passphrase</label>
              <div className="relative">
                <input
                  type={showPasswords['okx-pass'] ? 'text' : 'password'}
                  value={keys.okx.passphrase}
                  onChange={(e) => updateOKX({ passphrase: e.target.value })}
                  placeholder="OKX Passphrase Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('okx-pass')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['okx-pass'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={keys.okx.testnet}
                  onChange={(e) => updateOKX({ testnet: e.target.checked })}
                  className="w-4 h-4 text-yellow-600 bg-gray-700 border-gray-600 rounded"
                />
                <span className="text-sm text-gray-300">Testnet Kullan</span>
              </label>

              <button
                onClick={() => testConnection('okx')}
                disabled={testing === 'okx' || !keys.okx.apiKey || !keys.okx.secretKey || !keys.okx.passphrase}
                style={{
                  padding: '8px 16px',
                  background: testing === 'okx' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: '600',
                  fontSize: '13px',
                  cursor: testing === 'okx' ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  opacity: (!keys.okx.apiKey || !keys.okx.secretKey || !keys.okx.passphrase) ? 0.5 : 1,
                }}
              >
                {testing === 'okx' ? (
                  <Icons.Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Icons.Zap className="w-4 h-4" />
                )}
                <span>Bağlantıyı Test Et</span>
              </button>
            </div>

            {keys.okx.lastTested && (
              <div className="text-xs text-gray-500">
                Son test: {new Date(keys.okx.lastTested).toLocaleString()}
              </div>
            )}
          </div>
        </div>

        {/* 3. BYBIT */}
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">
              <Icons.TrendingUp style={{ color: '#F7A600' }} className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3>Bybit API</h3>
              <p className="text-gray-400 text-sm">Trading ve market data</p>
            </div>
            <div className="flex items-center gap-3">
              {getStatusIcon(keys.bybit.status)}
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={keys.bybit.enabled}
                  onChange={(e) => updateBybit({ enabled: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>

          <div className="settings-card-body">
            <div className="settings-form-group">
              <label>API Key</label>
              <div className="relative">
                <input
                  type={showPasswords['bybit-api'] ? 'text' : 'password'}
                  value={keys.bybit.apiKey}
                  onChange={(e) => updateBybit({ apiKey: e.target.value })}
                  placeholder="Bybit API Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('bybit-api')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['bybit-api'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="settings-form-group">
              <label>Secret Key</label>
              <div className="relative">
                <input
                  type={showPasswords['bybit-secret'] ? 'text' : 'password'}
                  value={keys.bybit.secretKey}
                  onChange={(e) => updateBybit({ secretKey: e.target.value })}
                  placeholder="Bybit Gizli Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('bybit-secret')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['bybit-secret'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={keys.bybit.testnet}
                  onChange={(e) => updateBybit({ testnet: e.target.checked })}
                  className="w-4 h-4 text-yellow-600 bg-gray-700 border-gray-600 rounded"
                />
                <span className="text-sm text-gray-300">Testnet Kullan</span>
              </label>

              <button
                onClick={() => testConnection('bybit')}
                disabled={testing === 'bybit' || !keys.bybit.apiKey || !keys.bybit.secretKey}
                style={{
                  padding: '8px 16px',
                  background: testing === 'bybit' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: '600',
                  fontSize: '13px',
                  cursor: testing === 'bybit' ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  opacity: (!keys.bybit.apiKey || !keys.bybit.secretKey) ? 0.5 : 1,
                }}
              >
                {testing === 'bybit' ? (
                  <Icons.Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Icons.Zap className="w-4 h-4" />
                )}
                <span>Bağlantıyı Test Et</span>
              </button>
            </div>

            {keys.bybit.lastTested && (
              <div className="text-xs text-gray-500">
                Son test: {new Date(keys.bybit.lastTested).toLocaleString()}
              </div>
            )}
          </div>
        </div>

        {/* 4. BTCTURK */}
        <div className="settings-premium-card">
          <div className="settings-card-header">
            <div className="settings-card-icon">
              <Icons.TrendingUp style={{ color: '#E30A17' }} className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <h3>BTCTurk API</h3>
              <p className="text-gray-400 text-sm">Türk Lirası piyasası</p>
            </div>
            <div className="flex items-center gap-3">
              {getStatusIcon(keys.btcturk.status)}
              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={keys.btcturk.enabled}
                  onChange={(e) => updateBTCTurk({ enabled: e.target.checked })}
                />
                <span className="toggle-slider"></span>
              </label>
            </div>
          </div>

          <div className="settings-card-body">
            <div className="settings-form-group">
              <label>API Key</label>
              <div className="relative">
                <input
                  type={showPasswords['btcturk-api'] ? 'text' : 'password'}
                  value={keys.btcturk.apiKey}
                  onChange={(e) => updateBTCTurk({ apiKey: e.target.value })}
                  placeholder="BTCTurk API Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('btcturk-api')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['btcturk-api'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="settings-form-group">
              <label>Secret Key</label>
              <div className="relative">
                <input
                  type={showPasswords['btcturk-secret'] ? 'text' : 'password'}
                  value={keys.btcturk.secretKey}
                  onChange={(e) => updateBTCTurk({ secretKey: e.target.value })}
                  placeholder="BTCTurk Gizli Anahtarını Girin"
                  className="settings-input-premium pr-10"
                />
                <button
                  onClick={() => togglePasswordVisibility('btcturk-secret')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                >
                  {showPasswords['btcturk-secret'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                onClick={() => testConnection('btcturk')}
                disabled={testing === 'btcturk' || !keys.btcturk.apiKey || !keys.btcturk.secretKey}
                style={{
                  padding: '8px 16px',
                  background: testing === 'btcturk' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                  color: '#FFFFFF',
                  border: 'none',
                  borderRadius: '6px',
                  fontWeight: '600',
                  fontSize: '13px',
                  cursor: testing === 'btcturk' ? 'not-allowed' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  opacity: (!keys.btcturk.apiKey || !keys.btcturk.secretKey) ? 0.5 : 1,
                }}
              >
                {testing === 'btcturk' ? (
                  <Icons.Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Icons.Zap className="w-4 h-4" />
                )}
                <span>Bağlantıyı Test Et</span>
              </button>
            </div>

            {keys.btcturk.lastTested && (
              <div className="text-xs text-gray-500">
                Son test: {new Date(keys.btcturk.lastTested).toLocaleString()}
              </div>
            )}
          </div>
        </div>

        {/* ADMIN-ONLY SECTION */}
        {isAdmin && (
          <>
            {/* 5. GROQ AI - ADMIN ONLY */}
            <div className="settings-premium-card" style={{ border: '2px solid #EF4444' }}>
              <div className="settings-card-header">
                <div className="settings-card-icon">
                  <Icons.Brain style={{ color: '#10B981' }} className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3>Groq AI API <span className="text-red-500 text-xs ml-2">(ADMIN)</span></h3>
                  <p className="text-gray-400 text-sm">AI Asistan ve analiz</p>
                </div>
                <div className="flex items-center gap-3">
                  {getStatusIcon(keys.groq.status)}
                  <label className="settings-toggle-premium">
                    <input
                      type="checkbox"
                      checked={keys.groq.enabled}
                      onChange={(e) => updateGroq({ enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              <div className="settings-card-body">
                <div className="settings-form-group">
                  <label>API Key</label>
                  <div className="relative">
                    <input
                      type={showPasswords['groq'] ? 'text' : 'password'}
                      value={keys.groq.apiKey}
                      onChange={(e) => updateGroq({ apiKey: e.target.value })}
                      placeholder="gsk_..."
                      className="settings-input-premium pr-10"
                    />
                    <button
                      onClick={() => togglePasswordVisibility('groq')}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                    >
                      {showPasswords['groq'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div className="settings-form-group">
                  <label>Model</label>
                  <select
                    value={keys.groq.model}
                    onChange={(e) => updateGroq({ model: e.target.value as any })}
                    className="settings-input-premium"
                  >
                    <option value="llama-3.3-70b-versatile">Llama 3.3 70B (Versatile)</option>
                    <option value="llama-3.1-8b-instant">Llama 3.1 8B (Instant)</option>
                    <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                  </select>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => testConnection('groq')}
                    disabled={testing === 'groq' || !keys.groq.apiKey}
                    style={{
                      padding: '8px 16px',
                      background: testing === 'groq' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '6px',
                      fontWeight: '600',
                      fontSize: '13px',
                      cursor: testing === 'groq' ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      opacity: !keys.groq.apiKey ? 0.5 : 1,
                    }}
                  >
                    {testing === 'groq' ? (
                      <Icons.Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Icons.Zap className="w-4 h-4" />
                    )}
                    <span>Bağlantıyı Test Et</span>
                  </button>
                </div>

                {keys.groq.lastTested && (
                  <div className="text-xs text-gray-500">
                    Son test: {new Date(keys.groq.lastTested).toLocaleString()}
                  </div>
                )}
              </div>
            </div>

            {/* 6. COINMARKETCAP - ADMIN ONLY */}
            <div className="settings-premium-card" style={{ border: '2px solid #EF4444' }}>
              <div className="settings-card-header">
                <div className="settings-card-icon">
                  <Icons.Wallet style={{ color: '#10B981' }} className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3>CoinMarketCap API <span className="text-red-500 text-xs ml-2">(ADMIN)</span></h3>
                  <p className="text-gray-400 text-sm">Kripto para piyasa verileri</p>
                </div>
                <div className="flex items-center gap-3">
                  {getStatusIcon(keys.coinmarketcap.status)}
                  <label className="settings-toggle-premium">
                    <input
                      type="checkbox"
                      checked={keys.coinmarketcap.enabled}
                      onChange={(e) => updateCoinMarketCap({ enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              <div className="settings-card-body">
                <div className="settings-form-group">
                  <label>API Key</label>
                  <div className="relative">
                    <input
                      type={showPasswords['cmc'] ? 'text' : 'password'}
                      value={keys.coinmarketcap.apiKey}
                      onChange={(e) => updateCoinMarketCap({ apiKey: e.target.value })}
                      placeholder="CoinMarketCap API Anahtarını Girin"
                      className="settings-input-premium pr-10"
                    />
                    <button
                      onClick={() => togglePasswordVisibility('cmc')}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                    >
                      {showPasswords['cmc'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div className="settings-form-group">
                  <label>Plan</label>
                  <select
                    value={keys.coinmarketcap.plan}
                    onChange={(e) => updateCoinMarketCap({ plan: e.target.value as any })}
                    className="settings-input-premium"
                  >
                    <option value="free">Ücretsiz (333 çağrı/gün)</option>
                    <option value="basic">Temel</option>
                    <option value="pro">Pro</option>
                  </select>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => testConnection('coinmarketcap')}
                    disabled={testing === 'coinmarketcap' || !keys.coinmarketcap.apiKey}
                    style={{
                      padding: '8px 16px',
                      background: testing === 'coinmarketcap' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '6px',
                      fontWeight: '600',
                      fontSize: '13px',
                      cursor: testing === 'coinmarketcap' ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      opacity: !keys.coinmarketcap.apiKey ? 0.5 : 1,
                    }}
                  >
                    {testing === 'coinmarketcap' ? (
                      <Icons.Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Icons.Zap className="w-4 h-4" />
                    )}
                    <span>Bağlantıyı Test Et</span>
                  </button>
                </div>

                {keys.coinmarketcap.lastTested && (
                  <div className="text-xs text-gray-500">
                    Son test: {new Date(keys.coinmarketcap.lastTested).toLocaleString()}
                  </div>
                )}
              </div>
            </div>

            {/* 7. TELEGRAM - ADMIN ONLY */}
            <div className="settings-premium-card" style={{ border: '2px solid #EF4444' }}>
              <div className="settings-card-header">
                <div className="settings-card-icon">
                  <Icons.Send style={{ color: '#10B981' }} className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3>Telegram Bot <span className="text-red-500 text-xs ml-2">(ADMIN)</span></h3>
                  <p className="text-gray-400 text-sm">Bildirimler ve uyarılar</p>
                </div>
                <div className="flex items-center gap-3">
                  {getStatusIcon(keys.telegram.status)}
                  <label className="settings-toggle-premium">
                    <input
                      type="checkbox"
                      checked={keys.telegram.enabled}
                      onChange={(e) => updateTelegram({ enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              <div className="settings-card-body">
                <div className="settings-form-group">
                  <label>Bot Token</label>
                  <div className="relative">
                    <input
                      type={showPasswords['telegram'] ? 'text' : 'password'}
                      value={keys.telegram.botToken}
                      onChange={(e) => updateTelegram({ botToken: e.target.value })}
                      placeholder="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
                      className="settings-input-premium pr-10"
                    />
                    <button
                      onClick={() => togglePasswordVisibility('telegram')}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                    >
                      {showPasswords['telegram'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div className="settings-form-group">
                  <label>Chat ID</label>
                  <input
                    type="text"
                    value={keys.telegram.chatId}
                    onChange={(e) => updateTelegram({ chatId: e.target.value })}
                    placeholder="Telegram Sohbet ID'niz"
                    className="settings-input-premium"
                  />
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => testConnection('telegram')}
                    disabled={testing === 'telegram' || !keys.telegram.botToken || !keys.telegram.chatId}
                    style={{
                      padding: '8px 16px',
                      background: testing === 'telegram' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '6px',
                      fontWeight: '600',
                      fontSize: '13px',
                      cursor: testing === 'telegram' ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      opacity: (!keys.telegram.botToken || !keys.telegram.chatId) ? 0.5 : 1,
                    }}
                  >
                    {testing === 'telegram' ? (
                      <Icons.Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Icons.Zap className="w-4 h-4" />
                    )}
                    <span>Bağlantıyı Test Et</span>
                  </button>
                </div>

                {keys.telegram.lastTested && (
                  <div className="text-xs text-gray-500">
                    Son test: {new Date(keys.telegram.lastTested).toLocaleString()}
                  </div>
                )}
              </div>
            </div>

            {/* 8. RAPIDAPI - ADMIN ONLY */}
            <div className="settings-premium-card" style={{ border: '2px solid #EF4444' }}>
              <div className="settings-card-header">
                <div className="settings-card-icon">
                  <Icons.Zap style={{ color: '#F59E0B' }} className="w-6 h-6" />
                </div>
                <div className="flex-1">
                  <h3>RapidAPI <span className="text-red-500 text-xs ml-2">(ADMIN)</span></h3>
                  <p className="text-gray-400 text-sm">Ek piyasa API'leri</p>
                </div>
                <div className="flex items-center gap-3">
                  {getStatusIcon(keys.rapidapi.status)}
                  <label className="settings-toggle-premium">
                    <input
                      type="checkbox"
                      checked={keys.rapidapi.enabled}
                      onChange={(e) => updateRapidAPI({ enabled: e.target.checked })}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              <div className="settings-card-body">
                <div className="settings-form-group">
                  <label>API Key</label>
                  <div className="relative">
                    <input
                      type={showPasswords['rapidapi'] ? 'text' : 'password'}
                      value={keys.rapidapi.apiKey}
                      onChange={(e) => updateRapidAPI({ apiKey: e.target.value })}
                      placeholder="RapidAPI Anahtarını Girin"
                      className="settings-input-premium pr-10"
                    />
                    <button
                      onClick={() => togglePasswordVisibility('rapidapi')}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white"
                    >
                      {showPasswords['rapidapi'] ? <Icons.EyeOff className="w-4 h-4" /> : <Icons.Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => testConnection('rapidapi')}
                    disabled={testing === 'rapidapi' || !keys.rapidapi.apiKey}
                    style={{
                      padding: '8px 16px',
                      background: testing === 'rapidapi' ? '#6B7280' : 'linear-gradient(135deg, #3B82F6, #1D4ED8)',
                      color: '#FFFFFF',
                      border: 'none',
                      borderRadius: '6px',
                      fontWeight: '600',
                      fontSize: '13px',
                      cursor: testing === 'rapidapi' ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      opacity: !keys.rapidapi.apiKey ? 0.5 : 1,
                    }}
                  >
                    {testing === 'rapidapi' ? (
                      <Icons.Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Icons.Zap className="w-4 h-4" />
                    )}
                    <span>Bağlantıyı Test Et</span>
                  </button>
                </div>

                {keys.rapidapi.lastTested && (
                  <div className="text-xs text-gray-500">
                    Son test: {new Date(keys.rapidapi.lastTested).toLocaleString()}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Auto-save indicator */}
      {saving && (
        <div className="fixed bottom-4 right-4 bg-black border-2 border-white rounded-lg px-4 py-2 flex items-center gap-2 shadow-lg">
          <Icons.Loader2 className="w-4 h-4 animate-spin text-white" />
          <span className="text-sm text-white">Kaydediliyor...</span>
        </div>
      )}
    </div>
  );
}
