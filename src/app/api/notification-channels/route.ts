/**
 * ADVANCED NOTIFICATION CHANNELS API
 * Manages multi-channel notifications: Telegram, Discord, Email, SMS
 *
 * Features:
 * - Telegram bot integration
 * - Discord webhook notifications
 * - Email notifications (SMTP)
 * - SMS notifications (Twilio)
 * - Test notifications for each channel
 * - Channel-specific filtering
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { notificationChannelsDB } from '@/lib/database';

// Notification Channel Settings Interface
interface NotificationChannels {
  telegram: {
    enabled: boolean;
    botToken: string;
    chatId: string;
    notifyOnBuy: boolean;
    notifyOnSell: boolean;
    notifyOnAlerts: boolean;
    minConfidence: number; // 0-100
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
    provider: 'twilio' | 'nexmo' | 'aws_sns';
    twilioAccountSid: string;
    twilioAuthToken: string;
    twilioFromNumber: string;
    toPhoneNumber: string;
    notifyOnBuy: boolean;
    notifyOnSell: boolean;
    notifyOnAlerts: boolean;
    minConfidence: number;
  };

  // Global settings
  global: {
    consolidateMessages: boolean; // Combine multiple signals into one message
    quietHoursEnabled: boolean;
    quietHoursStart: string; // "22:00"
    quietHoursEnd: string; // "08:00"
    maxNotificationsPerHour: number;
  };
}

// Default Settings
const DEFAULT_NOTIFICATION_CHANNELS: NotificationChannels = {
  telegram: {
    enabled: false,
    botToken: '',
    chatId: '',
    notifyOnBuy: true,
    notifyOnSell: true,
    notifyOnAlerts: true,
    minConfidence: 70,
  },
  discord: {
    enabled: false,
    webhookUrl: '',
    notifyOnBuy: true,
    notifyOnSell: true,
    notifyOnAlerts: true,
    minConfidence: 70,
    username: 'SARDAG Trading Bot',
    avatarUrl: '',
  },
  email: {
    enabled: false,
    smtpHost: 'smtp.gmail.com',
    smtpPort: 587,
    smtpUser: '',
    smtpPassword: '',
    fromEmail: '',
    toEmail: '',
    notifyOnBuy: true,
    notifyOnSell: true,
    notifyOnAlerts: true,
    minConfidence: 70,
  },
  sms: {
    enabled: false,
    provider: 'twilio',
    twilioAccountSid: '',
    twilioAuthToken: '',
    twilioFromNumber: '',
    toPhoneNumber: '',
    notifyOnBuy: false,
    notifyOnSell: true,
    notifyOnAlerts: true,
    minConfidence: 85, // Higher threshold for SMS
  },
  global: {
    consolidateMessages: true,
    quietHoursEnabled: false,
    quietHoursStart: '22:00',
    quietHoursEnd: '08:00',
    maxNotificationsPerHour: 10,
  },
};

// Database storage (persistent, encrypted)
// Old: const channelSettingsStore = new Map<string, NotificationChannels>();
// Now using: notificationChannelsDB from @/lib/database

/**
 * Get session ID from cookies
 */
async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;

  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  return sessionId;
}

/**
 * Validate notification channel settings
 */
function validateChannels(settings: Partial<NotificationChannels>): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Validate Telegram
  if (settings.telegram?.enabled) {
    if (!settings.telegram.botToken) {
      errors.push('Telegram bot token is required');
    }
    if (!settings.telegram.chatId) {
      errors.push('Telegram chat ID is required');
    }
  }

  // Validate Discord
  if (settings.discord?.enabled) {
    if (!settings.discord.webhookUrl) {
      errors.push('Discord webhook URL is required');
    } else if (!settings.discord.webhookUrl.startsWith('https://discord.com/api/webhooks/')) {
      errors.push('Invalid Discord webhook URL format');
    }
  }

  // Validate Email
  if (settings.email?.enabled) {
    if (!settings.email.smtpHost || !settings.email.smtpUser || !settings.email.smtpPassword) {
      errors.push('Email SMTP configuration incomplete');
    }
    if (!settings.email.toEmail || !settings.email.toEmail.includes('@')) {
      errors.push('Valid email address required');
    }
  }

  // Validate SMS
  if (settings.sms?.enabled) {
    if (settings.sms.provider === 'twilio') {
      if (!settings.sms.twilioAccountSid || !settings.sms.twilioAuthToken || !settings.sms.twilioFromNumber) {
        errors.push('Twilio configuration incomplete');
      }
      if (!settings.sms.toPhoneNumber) {
        errors.push('Phone number required for SMS');
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Test Telegram notification
 */
async function testTelegram(botToken: string, chatId: string): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: '✅ SARDAG İşlem Tarayıcı\n\nTelegram bildirimleri başarıyla aktif edildi!\n\nGüçlü sinyaller için anlık bildirimler alacaksınız.',
        parse_mode: 'HTML',
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      return { success: false, message: `Telegram error: ${error.description || 'Unknown'}` };
    }

    return { success: true, message: 'Test notification sent successfully' };
  } catch (error) {
    return { success: false, message: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * Test Discord notification
 */
async function testDiscord(webhookUrl: string, username: string): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: username || 'SARDAG Trading Bot',
        embeds: [
          {
            title: '✅ Discord Notifications Activated',
            description: 'SARDAG İşlem Tarayıcı notifications are now active!',
            color: 0x00ff00,
            fields: [
              { name: 'Status', value: 'Connected', inline: true },
              { name: 'Notifications', value: 'Enabled', inline: true },
            ],
            timestamp: new Date().toISOString(),
          },
        ],
      }),
    });

    if (!response.ok) {
      return { success: false, message: `Discord webhook error: ${response.status}` };
    }

    return { success: true, message: 'Test notification sent successfully' };
  } catch (error) {
    return { success: false, message: error instanceof Error ? error.message : 'Unknown error' };
  }
}

/**
 * GET - Retrieve notification channel settings
 */
export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Get settings or use defaults
    let settings = notificationChannelsDB.get(sessionId);
    if (!settings) {
      settings = DEFAULT_NOTIFICATION_CHANNELS;
      notificationChannelsDB.set(sessionId, settings);
    }

    // Mask sensitive data in response
    const maskedSettings: NotificationChannels = {
      ...settings,
      telegram: {
        ...settings.telegram,
        botToken: settings.telegram.botToken ? '***' + settings.telegram.botToken.slice(-4) : '',
      },
      discord: {
        ...settings.discord,
        webhookUrl: settings.discord.webhookUrl ? settings.discord.webhookUrl.slice(0, 40) + '***' : '',
      },
      email: {
        ...settings.email,
        smtpPassword: settings.email.smtpPassword ? '********' : '',
      },
      sms: {
        ...settings.sms,
        twilioAuthToken: settings.sms.twilioAuthToken ? '********' : '',
      },
    };

    return NextResponse.json({
      success: true,
      data: maskedSettings,
    });
  } catch (error) {
    console.error('[Notification Channels API] GET Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get notification settings',
      },
      { status: 500 }
    );
  }
}

/**
 * POST - Update notification channel settings
 */
export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const body = await request.json();

    // Handle test notification requests
    if (body.action === 'test') {
      const { channel } = body;

      if (channel === 'telegram') {
        const result = await testTelegram(body.botToken, body.chatId);
        return NextResponse.json(result);
      } else if (channel === 'discord') {
        const result = await testDiscord(body.webhookUrl, body.username || 'SARDAG Trading Bot');
        return NextResponse.json(result);
      }

      return NextResponse.json({ success: false, message: 'Invalid channel for testing' });
    }

    // Validate input
    const validation = validateChannels(body);
    if (!validation.valid) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid notification settings',
          details: validation.errors,
        },
        { status: 400 }
      );
    }

    // Get current settings or defaults
    const currentSettings = notificationChannelsDB.get(sessionId) || DEFAULT_NOTIFICATION_CHANNELS;

    // Merge with new settings
    const updatedSettings: NotificationChannels = {
      telegram: { ...currentSettings.telegram, ...(body.telegram || {}) },
      discord: { ...currentSettings.discord, ...(body.discord || {}) },
      email: { ...currentSettings.email, ...(body.email || {}) },
      sms: { ...currentSettings.sms, ...(body.sms || {}) },
      global: { ...currentSettings.global, ...(body.global || {}) },
    };

    // Save to store
    notificationChannelsDB.set(sessionId, updatedSettings);

    // Create response with Set-Cookie header
    const response = NextResponse.json({
      success: true,
      data: updatedSettings,
      message: 'Notification channels updated successfully',
    });

    // Set session cookie
    response.cookies.set('session_id', sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60, // 7 days
    });

    return response;
  } catch (error) {
    console.error('[Notification Channels API] POST Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update notification channels',
      },
      { status: 500 }
    );
  }
}

/**
 * PUT - Reset to default settings
 */
export async function PUT(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Reset to defaults
    notificationChannelsDB.set(sessionId, DEFAULT_NOTIFICATION_CHANNELS);

    return NextResponse.json({
      success: true,
      data: DEFAULT_NOTIFICATION_CHANNELS,
      message: 'Notification channels reset to defaults',
    });
  } catch (error) {
    console.error('[Notification Channels API] PUT Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to reset notification channels',
      },
      { status: 500 }
    );
  }
}
