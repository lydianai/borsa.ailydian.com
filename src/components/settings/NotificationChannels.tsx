"use client";

import { useState, useEffect } from "react";
import * as Icons from "lucide-react";

interface NotificationChannels {
  telegram: {
    enabled: boolean;
    botToken: string;
    chatId: string;
    notifyOnBuy: boolean;
    notifyOnSell: boolean;
    notifyOnAlerts: boolean;
    minConfidence: number;
  };
  discord: {
    enabled: boolean;
    webhookUrl: string;
    notifyOnBuy: boolean;
    notifyOnSell: boolean;
    notifyOnAlerts: boolean;
    minConfidence: number;
    username: string;
    avatarUrl: string;
  };
  email: {
    enabled: boolean;
    smtpHost: string;
    smtpPort: number;
    smtpUser: string;
    smtpPassword: string;
    fromEmail: string;
    toEmail: string;
    notifyOnBuy: boolean;
    notifyOnSell: boolean;
    notifyOnAlerts: boolean;
    minConfidence: number;
  };
  sms: {
    enabled: boolean;
    provider: "twilio" | "nexmo" | "aws_sns";
    twilioAccountSid: string;
    twilioAuthToken: string;
    twilioFromNumber: string;
    toPhoneNumber: string;
    notifyOnBuy: boolean;
    notifyOnSell: boolean;
    notifyOnAlerts: boolean;
    minConfidence: number;
  };
  global: {
    consolidateMessages: boolean;
    quietHoursEnabled: boolean;
    quietHoursStart: string;
    quietHoursEnd: string;
    maxNotificationsPerHour: number;
  };
}

interface NotificationChannelsProps {
  onSave?: () => void;
}

export default function NotificationChannels({ onSave }: NotificationChannelsProps) {
  const [settings, setSettings] = useState<NotificationChannels | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState<string | null>(null);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/notification-channels");
      const result = await response.json();

      if (result.success) {
        setSettings(result.data);
      }
    } catch (error) {
      console.error("Failed to load notification channels:", error);
      showMessage("error", "Ayarlar yüklenemedi");
    } finally {
      setLoading(false);
    }
  };

  const saveSettings = async () => {
    if (!settings) return;

    try {
      setSaving(true);
      const response = await fetch("/api/notification-channels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });

      const result = await response.json();

      if (result.success) {
        showMessage("success", "Bildirim kanalları kaydedildi");
        onSave?.();
      } else {
        showMessage("error", result.error || "Kaydetme başarısız");
      }
    } catch (error) {
      console.error("Failed to save notification channels:", error);
      showMessage("error", "Kaydetme hatası");
    } finally {
      setSaving(false);
    }
  };

  const testChannel = async (channel: "telegram" | "discord") => {
    if (!settings) return;

    try {
      setTesting(channel);

      const testData = channel === "telegram"
        ? {
            action: "test",
            channel: "telegram",
            botToken: settings.telegram.botToken,
            chatId: settings.telegram.chatId,
          }
        : {
            action: "test",
            channel: "discord",
            webhookUrl: settings.discord.webhookUrl,
            username: settings.discord.username,
          };

      const response = await fetch("/api/notification-channels", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testData),
      });

      const result = await response.json();

      if (result.success) {
        showMessage("success", `${channel === "telegram" ? "Telegram" : "Discord"} test bildirimi gönderildi!`);
      } else {
        showMessage("error", result.message || "Test başarısız");
      }
    } catch (error) {
      console.error(`Failed to test ${channel}:`, error);
      showMessage("error", "Test hatası");
    } finally {
      setTesting(null);
    }
  };

  const showMessage = (type: "success" | "error", text: string) => {
    setMessage({ type, text });
    setTimeout(() => setMessage(null), 3000);
  };

  const updateSettings = (channel: keyof NotificationChannels, field: string, value: any) => {
    if (!settings) return;

    setSettings({
      ...settings,
      [channel]: {
        ...settings[channel],
        [field]: value,
      },
    });
  };

  if (loading) {
    return (
      <div className="settings-content-wrapper">
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
        </div>
      </div>
    );
  }

  if (!settings) {
    return (
      <div className="settings-content-wrapper">
        <div className="settings-alert-error">
          <span>Bildirim kanalları yüklenemedi</span>
        </div>
      </div>
    );
  }

  return (
    <div className="settings-content-wrapper">
      {/* Message Banner */}
      {message && (
        <div className={message.type === "success" ? "settings-alert-success" : "settings-alert-error"}>
          <Icons.Bell className="w-5 h-5" />
          <span>{message.text}</span>
        </div>
      )}

      {/* Stats Grid */}
      <div className="settings-grid-3">
        <div className="settings-stat-card">
          <div className="stat-icon"><Icons.Bell style={{ color: '#F59E0B' }} className="w-8 h-8" /></div>
          <div className="stat-value">{Object.values(settings).filter((s: any) => s.enabled).length}</div>
          <div className="stat-label">Aktif Kanallar</div>
        </div>

        <div className="settings-stat-card">
          <div className="stat-icon"><Icons.Send style={{ color: '#3B82F6' }} className="w-6 h-6" /></div>
          <div className="stat-value">{settings.telegram.enabled ? 'AÇIK' : 'KAPALI'}</div>
          <div className="stat-label">Telegram</div>
        </div>

        <div className="settings-stat-card">
          <div className="stat-icon"><Icons.Mail style={{ color: '#10B981' }} className="w-6 h-6" /></div>
          <div className="stat-value">{settings.email.enabled ? 'AÇIK' : 'KAPALI'}</div>
          <div className="stat-label">E-posta</div>
        </div>
      </div>

      {/* Telegram Channel */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon"><Icons.Send style={{ color: '#3B82F6' }} className="w-6 h-6" /></div>
          <h3>Telegram Bildirimleri</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.telegram.enabled}
              onChange={(e) => updateSettings("telegram", "enabled", e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Telegram'ı Aktif Et</span>
          </label>

          {settings.telegram.enabled && (
            <>
              <div className="settings-alert-info">
                <span className="text-sm">
                  <strong>Telegram Bot Kurulumu:</strong><br />
                  1. @BotFather ile konuş ve /newbot komutunu kullan<br />
                  2. Bot token'ını al ve aşağıya yapıştır<br />
                  3. Botuna /start mesajı gönder<br />
                  4. @userinfobot ile chat ID'ni öğren
                </span>
              </div>

              <div className="settings-form-group">
                <label>Bot Token</label>
                <input
                  type="password"
                  value={settings.telegram.botToken}
                  onChange={(e) => updateSettings("telegram", "botToken", e.target.value)}
                  placeholder="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Chat ID</label>
                <input
                  type="text"
                  value={settings.telegram.chatId}
                  onChange={(e) => updateSettings("telegram", "chatId", e.target.value)}
                  placeholder="123456789"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Minimum Güven Seviyesi: {settings.telegram.minConfidence}%</label>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={settings.telegram.minConfidence}
                  onChange={(e) => updateSettings("telegram", "minConfidence", parseInt(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>

              <div className="space-y-2">
                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.telegram.notifyOnBuy}
                    onChange={(e) => updateSettings("telegram", "notifyOnBuy", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">AL sinyalleri için bildir</span>
                </label>

                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.telegram.notifyOnSell}
                    onChange={(e) => updateSettings("telegram", "notifyOnSell", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">SAT sinyalleri için bildir</span>
                </label>

                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.telegram.notifyOnAlerts}
                    onChange={(e) => updateSettings("telegram", "notifyOnAlerts", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">Uyarılar için bildir</span>
                </label>
              </div>

              <button
                onClick={() => testChannel("telegram")}
                disabled={testing === "telegram" || !settings.telegram.botToken || !settings.telegram.chatId}
                className="settings-btn-primary w-full"
              >
                <Icons.TestTube className="w-4 h-4" />
                <span>{testing === "telegram" ? "Test ediliyor..." : "Test Bildirimi Gönder"}</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Discord Channel */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon"><Icons.MessageSquare className="w-5 h-5" /></div>
          <h3>Discord Bildirimleri</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.discord.enabled}
              onChange={(e) => updateSettings("discord", "enabled", e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Discord'u Aktif Et</span>
          </label>

          {settings.discord.enabled && (
            <>
              <div className="settings-alert-info">
                <span className="text-sm">
                  <strong>Discord Webhook Kurulumu:</strong><br />
                  1. Sunucu Ayarları → Entegrasyonlar → Webhook Oluştur<br />
                  2. Kanal seç ve Webhook URL'sini kopyala<br />
                  3. URL'yi aşağıya yapıştır
                </span>
              </div>

              <div className="settings-form-group">
                <label>Webhook URL</label>
                <input
                  type="password"
                  value={settings.discord.webhookUrl}
                  onChange={(e) => updateSettings("discord", "webhookUrl", e.target.value)}
                  placeholder="https://discord.com/api/webhooks/..."
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Bot Kullanıcı Adı</label>
                <input
                  type="text"
                  value={settings.discord.username}
                  onChange={(e) => updateSettings("discord", "username", e.target.value)}
                  placeholder="LyTrade Trading Bot"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Minimum Güven Seviyesi: {settings.discord.minConfidence}%</label>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={settings.discord.minConfidence}
                  onChange={(e) => updateSettings("discord", "minConfidence", parseInt(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>

              <div className="space-y-2">
                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.discord.notifyOnBuy}
                    onChange={(e) => updateSettings("discord", "notifyOnBuy", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">AL sinyalleri için bildir</span>
                </label>

                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.discord.notifyOnSell}
                    onChange={(e) => updateSettings("discord", "notifyOnSell", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">SAT sinyalleri için bildir</span>
                </label>

                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.discord.notifyOnAlerts}
                    onChange={(e) => updateSettings("discord", "notifyOnAlerts", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">Uyarılar için bildir</span>
                </label>
              </div>

              <button
                onClick={() => testChannel("discord")}
                disabled={testing === "discord" || !settings.discord.webhookUrl}
                className="settings-btn-primary w-full"
              >
                <Icons.TestTube className="w-4 h-4" />
                <span>{testing === "discord" ? "Test ediliyor..." : "Test Bildirimi Gönder"}</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Email Channel */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon"><Icons.Mail style={{ color: '#10B981' }} className="w-6 h-6" /></div>
          <h3>E-posta Bildirimleri</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.email.enabled}
              onChange={(e) => updateSettings("email", "enabled", e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Email'i Aktif Et</span>
          </label>

          {settings.email.enabled && (
            <>
              <div className="settings-grid-2">
                <div className="settings-form-group">
                  <label>SMTP Host</label>
                  <input
                    type="text"
                    value={settings.email.smtpHost}
                    onChange={(e) => updateSettings("email", "smtpHost", e.target.value)}
                    placeholder="smtp.gmail.com"
                    className="settings-input-premium"
                  />
                </div>

                <div className="settings-form-group">
                  <label>Port</label>
                  <input
                    type="number"
                    value={settings.email.smtpPort}
                    onChange={(e) => updateSettings("email", "smtpPort", parseInt(e.target.value))}
                    className="settings-input-premium"
                  />
                </div>
              </div>

              <div className="settings-form-group">
                <label>SMTP Kullanıcı</label>
                <input
                  type="text"
                  value={settings.email.smtpUser}
                  onChange={(e) => updateSettings("email", "smtpUser", e.target.value)}
                  placeholder="your-email@gmail.com"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>SMTP Şifre</label>
                <input
                  type="password"
                  value={settings.email.smtpPassword}
                  onChange={(e) => updateSettings("email", "smtpPassword", e.target.value)}
                  placeholder="App Password"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Alıcı Email</label>
                <input
                  type="email"
                  value={settings.email.toEmail}
                  onChange={(e) => updateSettings("email", "toEmail", e.target.value)}
                  placeholder="recipient@example.com"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Minimum Güven Seviyesi: {settings.email.minConfidence}%</label>
                <input
                  type="range"
                  min="50"
                  max="100"
                  value={settings.email.minConfidence}
                  onChange={(e) => updateSettings("email", "minConfidence", parseInt(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* SMS Channel */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon"><Icons.Smartphone style={{ color: '#8B5CF6' }} className="w-6 h-6" /></div>
          <h3>SMS Bildirimleri</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.sms.enabled}
              onChange={(e) => updateSettings("sms", "enabled", e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">SMS'i Aktif Et</span>
          </label>

          {settings.sms.enabled && (
            <>
              <div className="settings-alert-warning">
                <span className="text-sm">
                  <strong>Önemli:</strong> SMS bildirimleri ücretlidir. Sadece yüksek güven seviyeli kritik sinyaller için kullanın.
                </span>
              </div>

              <div className="settings-form-group">
                <label>Twilio Account SID</label>
                <input
                  type="password"
                  value={settings.sms.twilioAccountSid}
                  onChange={(e) => updateSettings("sms", "twilioAccountSid", e.target.value)}
                  placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Twilio Auth Token</label>
                <input
                  type="password"
                  value={settings.sms.twilioAuthToken}
                  onChange={(e) => updateSettings("sms", "twilioAuthToken", e.target.value)}
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Gönderen Numara (Twilio)</label>
                <input
                  type="tel"
                  value={settings.sms.twilioFromNumber}
                  onChange={(e) => updateSettings("sms", "twilioFromNumber", e.target.value)}
                  placeholder="+1234567890"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Alıcı Numara</label>
                <input
                  type="tel"
                  value={settings.sms.toPhoneNumber}
                  onChange={(e) => updateSettings("sms", "toPhoneNumber", e.target.value)}
                  placeholder="+905xxxxxxxxx"
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Minimum Güven Seviyesi: {settings.sms.minConfidence}%</label>
                <input
                  type="range"
                  min="70"
                  max="100"
                  value={settings.sms.minConfidence}
                  onChange={(e) => updateSettings("sms", "minConfidence", parseInt(e.target.value))}
                  className="settings-slider-premium"
                />
              </div>

              <div className="space-y-2">
                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.sms.notifyOnSell}
                    onChange={(e) => updateSettings("sms", "notifyOnSell", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">SAT sinyalleri için bildir</span>
                </label>

                <label className="settings-toggle-premium">
                  <input
                    type="checkbox"
                    checked={settings.sms.notifyOnAlerts}
                    onChange={(e) => updateSettings("sms", "notifyOnAlerts", e.target.checked)}
                  />
                  <span className="toggle-slider"></span>
                  <span className="toggle-label">Kritik uyarılar için bildir</span>
                </label>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Global Settings */}
      <div className="settings-premium-card">
        <div className="settings-card-header">
          <div className="settings-card-icon"><Icons.Settings className="w-5 h-5" /></div>
          <h3>Genel Ayarlar</h3>
        </div>

        <div className="space-y-4">
          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.global.consolidateMessages}
              onChange={(e) => updateSettings("global", "consolidateMessages", e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Mesajları birleştir (aynı anda birden fazla sinyal varsa)</span>
          </label>

          <label className="settings-toggle-premium">
            <input
              type="checkbox"
              checked={settings.global.quietHoursEnabled}
              onChange={(e) => updateSettings("global", "quietHoursEnabled", e.target.checked)}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-label">Sessiz saatler (bildirim almayacağınız saatler)</span>
          </label>

          {settings.global.quietHoursEnabled && (
            <div className="settings-grid-2">
              <div className="settings-form-group">
                <label>Başlangıç</label>
                <input
                  type="time"
                  value={settings.global.quietHoursStart}
                  onChange={(e) => updateSettings("global", "quietHoursStart", e.target.value)}
                  className="settings-input-premium"
                />
              </div>

              <div className="settings-form-group">
                <label>Bitiş</label>
                <input
                  type="time"
                  value={settings.global.quietHoursEnd}
                  onChange={(e) => updateSettings("global", "quietHoursEnd", e.target.value)}
                  className="settings-input-premium"
                />
              </div>
            </div>
          )}

          <div className="settings-form-group">
            <label>Saatlik maksimum bildirim: {settings.global.maxNotificationsPerHour}</label>
            <input
              type="range"
              min="1"
              max="50"
              value={settings.global.maxNotificationsPerHour}
              onChange={(e) => updateSettings("global", "maxNotificationsPerHour", parseInt(e.target.value))}
              className="settings-slider-premium"
            />
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex gap-4">
        <button
          onClick={saveSettings}
          disabled={saving}
          className="settings-btn-primary flex-1"
        >
          <Icons.Check className="w-4 h-4" />
          <span>{saving ? "Kaydediliyor..." : "Ayarları Kaydet"}</span>
        </button>
      </div>
    </div>
  );
}
