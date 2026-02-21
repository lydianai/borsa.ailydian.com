"use client";

import { useState, useEffect } from "react";
import * as Icons from "lucide-react";

interface RiskSettings {
  globalStopLoss: {
    enabled: boolean;
    percentage: number;
    trailingStop: boolean;
    trailingDistance: number;
  };
  dailyLimits: {
    enabled: boolean;
    maxDailyLoss: number;
    maxDailyTrades: number;
    pauseTradingOnLimit: boolean;
  };
  positionSizing: {
    method: "fixed" | "percentage" | "kelly";
    fixedAmount: number;
    portfolioPercentage: number;
    maxPositionSize: number;
  };
  riskReward: {
    minRatio: number;
    autoCalculate: boolean;
  };
  autoClose: {
    enabled: boolean;
    takeProfit: number;
    stopLoss: number;
  };
  alerts: {
    notifyOnStopLoss: boolean;
    notifyOnDailyLimit: boolean;
    notifyOnLargePosition: boolean;
  };
}

interface DailyStats {
  date: string;
  totalLoss: number;
  totalProfit: number;
  tradesCount: number;
}

interface RiskManagementProps {
  onSave?: () => void;
}

export default function RiskManagement({ onSave }: RiskManagementProps) {
  const [settings, setSettings] = useState<RiskSettings | null>(null);
  const [dailyStats, setDailyStats] = useState<DailyStats | null>(null);
  const [limitsReached, setLimitsReached] = useState({ dailyLoss: false, dailyTrades: false });
  const [tradingPaused, setTradingPaused] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Position calculator state
  const [calculatorInputs, setCalculatorInputs] = useState({
    portfolioValue: 10000,
    entryPrice: 0,
    stopLossPrice: 0,
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/risk-management");
      const result = await response.json();

      if (result.success) {
        setSettings(result.data.settings);
        setDailyStats(result.data.dailyStats);
        setLimitsReached(result.data.limitsReached);
        setTradingPaused(result.data.tradingPaused);
      }
    } catch (error) {
      console.error("Failed to load risk settings:", error);
      showMessage("error", "Ayarlar yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    if (!settings) return;

    try {
      setSaving(true);
      const response = await fetch("/api/risk-management", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });

      const result = await response.json();

      if (result.success) {
        showMessage("success", "Risk ayarları kaydedildi");
        onSave?.();
      } else {
        showMessage("error", result.error || "Kaydetme başarısız");
      }
    } catch (error) {
      console.error("Failed to save risk settings:", error);
      showMessage("error", "Kaydetme hatası");
    } finally {
      setSaving(false);
    }
  };

  const resetSettings = async () => {
    if (!confirm("Tüm risk ayarlarını varsayılana sıfırlamak istediğinize emin misiniz?")) {
      return;
    }

    try {
      setSaving(true);
      const response = await fetch("/api/risk-management", {
        method: "PUT",
      });

      const result = await response.json();

      if (result.success) {
        setSettings(result.data);
        showMessage("success", "Ayarlar sıfırlandı");
      }
    } catch (error) {
      console.error("Failed to reset settings:", error);
      showMessage("error", "Sıfırlama başarısız");
    } finally {
      setSaving(false);
    }
  };

  const showMessage = (type: "success" | "error", text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  const updateSettings = (path: string[], value: any) => {
    if (!settings) return;

    const newSettings = { ...settings };
    let current: any = newSettings;

    for (let i = 0; i < path.length - 1; i++) {
      current[path[i]] = { ...current[path[i]] };
      current = current[path[i]];
    }

    current[path[path.length - 1]] = value;
    setSettings(newSettings);
  };

  const calculatePositionSize = () => {
    if (!settings) return { shares: 0, riskAmount: 0, positionValue: 0 };

    const { portfolioValue, entryPrice, stopLossPrice } = calculatorInputs;

    if (entryPrice <= 0 || stopLossPrice <= 0 || portfolioValue <= 0) {
      return { shares: 0, riskAmount: 0, positionValue: 0 };
    }

    const riskPerShare = Math.abs(entryPrice - stopLossPrice);
    const riskPercentage = settings.globalStopLoss.percentage / 100;
    const maxRiskAmount = portfolioValue * riskPercentage;
    let shares = Math.floor(maxRiskAmount / riskPerShare);
    let positionValue = shares * entryPrice;

    if (settings.positionSizing.method === "fixed") {
      positionValue = Math.min(positionValue, settings.positionSizing.fixedAmount);
      shares = Math.floor(positionValue / entryPrice);
    } else if (settings.positionSizing.method === "percentage") {
      const maxPosition = portfolioValue * (settings.positionSizing.portfolioPercentage / 100);
      positionValue = Math.min(positionValue, maxPosition);
      shares = Math.floor(positionValue / entryPrice);
    }

    if (positionValue > settings.positionSizing.maxPositionSize) {
      positionValue = settings.positionSizing.maxPositionSize;
      shares = Math.floor(positionValue / entryPrice);
    }

    const riskAmount = shares * riskPerShare;
    return { shares, riskAmount, positionValue };
  };

  const positionCalc = calculatePositionSize();

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="text-center p-8 text-red-500">
        Risk ayarları yüklenemedi
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Message Banner */}
      {message && (
        <div className={message.type === "success" ? "settings-alert-success" : "settings-alert-warning"}>
          {(Icons.AlertTriangle as any)({ style: { color: '#EF4444' }, size: 20 })}
          <span>{message.text}</span>
        </div>
      )}

      {/* Trading Status Alert */}
      {tradingPaused && (
        <div className="settings-alert-warning">
          {(Icons.AlertTriangle as any)({ style: { color: '#EF4444' }, size: 20 })}
          <div>
            <div className="font-bold">İşlem Durduruldu!</div>
            <div className="text-sm">Günlük limit aşıldı. Yarın otomatik sıfırlanacak.</div>
          </div>
        </div>
      )}

      {/* Daily Stats */}
      {dailyStats && (
        <div className="settings-grid-3">
          <div className="settings-stat-card">
            <div className="stat-icon">{(Icons.AlertTriangle as any)({ style: { color: '#EF4444' }, size: 32 })}</div>
            <div className="stat-value">${dailyStats.totalLoss.toFixed(2)}</div>
            <div className="stat-label">Günlük Zarar</div>
            {settings.dailyLimits.enabled && (
              <div className="text-xs text-gray-500 mt-1">
                Limit: ${settings.dailyLimits.maxDailyLoss}
              </div>
            )}
          </div>

          <div className="settings-stat-card">
            <div className="stat-icon">{(Icons.DollarSign as any)({ style: { color: '#10B981' }, size: 32 })}</div>
            <div className="stat-value">${dailyStats.totalProfit.toFixed(2)}</div>
            <div className="stat-label">Günlük Kar</div>
          </div>

          <div className="settings-stat-card">
            <div className="stat-icon">{(Icons.Shield as any)({ style: { color: '#F59E0B' }, size: 32 })}</div>
            <div className="stat-value">{dailyStats.tradesCount}</div>
            <div className="stat-label">İşlem Sayısı</div>
            {settings.dailyLimits.enabled && (
              <div className="text-xs text-gray-500 mt-1">
                Limit: {settings.dailyLimits.maxDailyTrades}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Stop-Loss Settings */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.Shield as any)({ style: { color: '#F59E0B' }, size: 24 })}</div>
          <h3>Zarar Durdur Ayarları</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.globalStopLoss.enabled}
              onChange={(e) => updateSettings(["globalStopLoss", "enabled"], e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Genel Zarar Durdur Aktif</span>
          </label>

          {settings.globalStopLoss.enabled && (
            <>
              <div className="settings-form-group">
                <label className="settings-label">
                  <span className="label-text">Zarar Durdur Yüzdesi: {settings.globalStopLoss.percentage}%</span>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="20"
                  step="0.5"
                  value={settings.globalStopLoss.percentage}
                  onChange={(e) => updateSettings(["globalStopLoss", "percentage"], parseFloat(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>

              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={settings.globalStopLoss.trailingStop}
                  onChange={(e) => updateSettings(["globalStopLoss", "trailingStop"], e.target.checked)}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">İzleyen Zarar Durdur Aktif</span>
              </label>

              {settings.globalStopLoss.trailingStop && (
                <div className="settings-form-group">
                  <label className="settings-label">
                    <span className="label-text">İzleme Mesafesi: {settings.globalStopLoss.trailingDistance}%</span>
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="10"
                    step="0.1"
                    value={settings.globalStopLoss.trailingDistance}
                    onChange={(e) => updateSettings(["globalStopLoss", "trailingDistance"], parseFloat(e.target.value))}
                    className="settings-slider-premium"
                  />
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Daily Limits */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.AlertTriangle as any)({ style: { color: '#EF4444' }, size: 24 })}</div>
          <h3>Günlük Limitler</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.dailyLimits.enabled}
              onChange={(e) => updateSettings(["dailyLimits", "enabled"], e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Günlük Limitler Aktif</span>
          </label>

          {settings.dailyLimits.enabled && (
            <>
              <div className="settings-form-group">
                <label className="settings-label">
                  {(Icons.DollarSign as any)({ size: 16, className: "label-icon" })}
                  <span className="label-text">Maksimum Günlük Zarar: ${settings.dailyLimits.maxDailyLoss}</span>
                </label>
                <input
                  type="number"
                  min="10"
                  max="1000000"
                  value={settings.dailyLimits.maxDailyLoss}
                  onChange={(e) => updateSettings(["dailyLimits", "maxDailyLoss"], parseInt(e.target.value) || 10)}
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label className="settings-label">
                  <span className="label-text">Maksimum İşlem Sayısı: {settings.dailyLimits.maxDailyTrades}</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="1000"
                  value={settings.dailyLimits.maxDailyTrades}
                  onChange={(e) => updateSettings(["dailyLimits", "maxDailyTrades"], parseInt(e.target.value) || 1)}
                  className="settings-input-premium"
                />
              </div>

              <label className="settings-toggle-premium">
                <input
                  type="checkbox"
                  checked={settings.dailyLimits.pauseTradingOnLimit}
                  onChange={(e) => updateSettings(["dailyLimits", "pauseTradingOnLimit"], e.target.checked)}
                />
                <span className="toggle-slider"></span>
                <span className="toggle-label">Limit aşıldığında işlemi durdur</span>
              </label>
            </>
          )}
        </div>
      </div>

      {/* Position Sizing */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.Percent as any)({ style: { color: '#3B82F6' }, size: 24 })}</div>
          <h3>Pozisyon Büyüklüğü</h3>
        </div>

        <div className="space-y-4">
          <div className="settings-form-group">
            <label className="settings-label">Hesaplama Yöntemi</label>
            <select
              value={settings.positionSizing.method}
              onChange={(e) => updateSettings(["positionSizing", "method"], e.target.value)}
              className="settings-input-premium"
            >
              <option value="fixed">Sabit Tutar</option>
              <option value="percentage">Portföy Yüzdesi</option>
              <option value="kelly">Kelly Criterion</option>
            </select>
          </div>

          {settings.positionSizing.method === "fixed" && (
            <div className="settings-form-group">
              <label className="settings-label">Sabit Tutar: ${settings.positionSizing.fixedAmount}</label>
              <input
                type="number"
                min="10"
                value={settings.positionSizing.fixedAmount}
                onChange={(e) => updateSettings(["positionSizing", "fixedAmount"], parseInt(e.target.value) || 10)}
                className="settings-input-premium"
              />
            </div>
          )}

          {settings.positionSizing.method === "percentage" && (
            <div className="settings-form-group">
              <label className="settings-label">Portföy Yüzdesi: {settings.positionSizing.portfolioPercentage}%</label>
              <input
                type="range"
                min="0.1"
                max="100"
                step="0.1"
                value={settings.positionSizing.portfolioPercentage}
                onChange={(e) => updateSettings(["positionSizing", "portfolioPercentage"], parseFloat(e.target.value))}
                className="settings-slider-premium"
              />
            </div>
          )}

          <div className="settings-form-group">
            <label className="settings-label">Maksimum Pozisyon: ${settings.positionSizing.maxPositionSize}</label>
            <input
              type="number"
              min="10"
              value={settings.positionSizing.maxPositionSize}
              onChange={(e) => updateSettings(["positionSizing", "maxPositionSize"], parseInt(e.target.value) || 10)}
              className="settings-input-premium"
            />
          </div>
        </div>
      </div>

      {/* Position Calculator */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.Calculator as any)({ style: { color: '#8B5CF6' }, size: 24 })}</div>
          <h3>Pozisyon Hesaplayıcı</h3>
        </div>

        <div className="settings-grid-3 mb-4">
          <div className="settings-form-group">
            <label className="settings-label">Portföy Değeri ($)</label>
            <input
              type="number"
              value={calculatorInputs.portfolioValue}
              onChange={(e) => setCalculatorInputs({ ...calculatorInputs, portfolioValue: parseFloat(e.target.value) || 0 })}
              className="settings-input-premium"
            />
          </div>

          <div className="settings-form-group">
            <label className="settings-label">Giriş Fiyatı ($)</label>
            <input
              type="number"
              step="0.01"
              value={calculatorInputs.entryPrice}
              onChange={(e) => setCalculatorInputs({ ...calculatorInputs, entryPrice: parseFloat(e.target.value) || 0 })}
              className="settings-input-premium"
            />
          </div>

          <div className="settings-form-group">
            <label className="settings-label">Zarar Durdur Fiyatı ($)</label>
            <input
              type="number"
              step="0.01"
              value={calculatorInputs.stopLossPrice}
              onChange={(e) => setCalculatorInputs({ ...calculatorInputs, stopLossPrice: parseFloat(e.target.value) || 0 })}
              className="settings-input-premium"
            />
          </div>
        </div>

        <div className="settings-grid-3">
          <div className="settings-stat-card">
            <div className="stat-label">Alınacak Miktar</div>
            <div className="stat-value">{positionCalc.shares.toFixed(2)}</div>
          </div>
          <div className="settings-stat-card">
            <div className="stat-label">Pozisyon Değeri</div>
            <div className="stat-value">${positionCalc.positionValue.toFixed(2)}</div>
          </div>
          <div className="settings-stat-card">
            <div className="stat-label">Risk Tutarı</div>
            <div className="stat-value">${positionCalc.riskAmount.toFixed(2)}</div>
          </div>
        </div>
      </div>

      {/* Auto-Close Settings */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.Shield as any)({ style: { color: '#F59E0B' }, size: 24 })}</div>
          <h3>Otomatik Kapanış</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.autoClose.enabled}
              onChange={(e) => updateSettings(["autoClose", "enabled"], e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Otomatik Kapanış Aktif</span>
          </label>

          {settings.autoClose.enabled && (
            <>
              <div className="settings-form-group">
                <label className="settings-label">Kar Al: {settings.autoClose.takeProfit}%</label>
                <input
                  type="range"
                  min="1"
                  max="50"
                  step="0.5"
                  value={settings.autoClose.takeProfit}
                  onChange={(e) => updateSettings(["autoClose", "takeProfit"], parseFloat(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>

              <div className="settings-form-group">
                <label className="settings-label">Zarar Durdur: {settings.autoClose.stopLoss}%</label>
                <input
                  type="range"
                  min="0.5"
                  max="20"
                  step="0.5"
                  value={settings.autoClose.stopLoss}
                  onChange={(e) => updateSettings(["autoClose", "stopLoss"], parseFloat(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Alerts */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon">{(Icons.Bell as any)({ style: { color: '#06B6D4' }, size: 24 })}</div>
          <h3>Bildirimler</h3>
        </div>

        <div className="space-y-2">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.alerts.notifyOnStopLoss}
              onChange={(e) => updateSettings(["alerts", "notifyOnStopLoss"], e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Zarar Durdur tetiklendiğinde bildir</span>
          </label>

          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.alerts.notifyOnDailyLimit}
              onChange={(e) => updateSettings(["alerts", "notifyOnDailyLimit"], e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Günlük limit aşıldığında bildir</span>
          </label>

          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.alerts.notifyOnLargePosition}
              onChange={(e) => updateSettings(["alerts", "notifyOnLargePosition"], e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Büyük pozisyon açıldığında bildir</span>
          </label>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={saveSettings}
          disabled={saving}
          className="settings-btn-primary flex-1"
        >
          {saving ? "Kaydediliyor..." : "Ayarları Kaydet"}
        </button>

        <button
          onClick={resetSettings}
          disabled={saving}
          className="settings-btn-secondary"
        >
          Sıfırla
        </button>
      </div>
    </div>
  );
}
