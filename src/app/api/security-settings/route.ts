/**
 * SECURITY SETTINGS API
 * 2FA, IP Whitelist, Session Management, Login History
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { securitySettingsDB } from '@/lib/database';

interface SecuritySettings {
  twoFactorAuth: {
    enabled: boolean;
    method: 'totp' | 'sms' | 'email';
    secret: string | null; // TOTP secret
    backupCodes: string[];
    lastVerified: string | null;
  };

  ipWhitelist: {
    enabled: boolean;
    allowedIPs: string[];
    blockUnknown: boolean;
    currentIP: string;
  };

  sessionManagement: {
    maxActiveSessions: number; // Max concurrent sessions
    sessionTimeout: number; // Minutes
    autoLogoutOnInactive: boolean;
    inactivityTimeout: number; // Minutes
    requireReauthForSensitive: boolean;
  };

  loginHistory: {
    enabled: boolean;
    keepDays: number;
    notifyOnNewDevice: boolean;
    notifyOnUnusualLocation: boolean;
    history: {
      timestamp: string;
      ip: string;
      location: string;
      device: string;
      success: boolean;
    }[];
  };

  apiSecurity: {
    rateLimiting: {
      enabled: boolean;
      maxRequests: number; // Per minute
      burstLimit: number;
    };
    allowedOrigins: string[];
    requireAPIKey: boolean;
  };

  passwordPolicy: {
    minLength: number;
    requireUppercase: boolean;
    requireLowercase: boolean;
    requireNumbers: boolean;
    requireSpecialChars: boolean;
    expiryDays: number; // 0 = never
    preventReuse: number; // Last N passwords
  };

  activityLogs: {
    enabled: boolean;
    logTypes: ('login' | 'settings_change' | 'api_call' | 'trade_action')[];
    retentionDays: number;
  };
}

const DEFAULT_SECURITY_SETTINGS: SecuritySettings = {
  twoFactorAuth: {
    enabled: false,
    method: 'totp',
    secret: null,
    backupCodes: [],
    lastVerified: null,
  },
  ipWhitelist: {
    enabled: false,
    allowedIPs: [],
    blockUnknown: false,
    currentIP: '0.0.0.0',
  },
  sessionManagement: {
    maxActiveSessions: 3,
    sessionTimeout: 720, // 12 hours
    autoLogoutOnInactive: true,
    inactivityTimeout: 30,
    requireReauthForSensitive: true,
  },
  loginHistory: {
    enabled: true,
    keepDays: 30,
    notifyOnNewDevice: true,
    notifyOnUnusualLocation: true,
    history: [
      {
        timestamp: new Date().toISOString(),
        ip: '192.168.1.1',
        location: 'Istanbul, Turkey',
        device: 'Chrome 120 on macOS',
        success: true,
      },
      {
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        ip: '192.168.1.1',
        location: 'Istanbul, Turkey',
        device: 'Chrome 120 on macOS',
        success: true,
      },
    ],
  },
  apiSecurity: {
    rateLimiting: {
      enabled: true,
      maxRequests: 100,
      burstLimit: 200,
    },
    allowedOrigins: ['http://localhost:3000'],
    requireAPIKey: false,
  },
  passwordPolicy: {
    minLength: 8,
    requireUppercase: true,
    requireLowercase: true,
    requireNumbers: true,
    requireSpecialChars: false,
    expiryDays: 0,
    preventReuse: 3,
  },
  activityLogs: {
    enabled: true,
    logTypes: ['login', 'settings_change', 'api_call'],
    retentionDays: 30,
  },
};

// Database storage (persistent, encrypted)
// Old: const securitySettingsStore = new Map<string, SecuritySettings>();
// Now using: securitySettingsDB from @/lib/database

async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  return sessionId;
}

function getClientIP(request: NextRequest): string {
  return (
    request.headers.get('x-forwarded-for')?.split(',')[0] ||
    request.headers.get('x-real-ip') ||
    '0.0.0.0'
  );
}

function generateBackupCodes(): string[] {
  const codes: string[] = [];
  for (let i = 0; i < 10; i++) {
    const code = Math.random().toString(36).substr(2, 8).toUpperCase();
    codes.push(`${code.substr(0, 4)}-${code.substr(4, 4)}`);
  }
  return codes;
}

function validateSecuritySettings(settings: Partial<SecuritySettings>): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (settings.ipWhitelist?.allowedIPs) {
    const ipRegex = /^(\d{1,3}\.){3}\d{1,3}$/;
    settings.ipWhitelist.allowedIPs.forEach(ip => {
      if (!ipRegex.test(ip)) {
        errors.push(`Invalid IP address: ${ip}`);
      }
    });
  }

  if (settings.sessionManagement) {
    if (settings.sessionManagement.maxActiveSessions < 1 || settings.sessionManagement.maxActiveSessions > 10) {
      errors.push('Max active sessions must be between 1 and 10');
    }
    if (settings.sessionManagement.sessionTimeout < 5 || settings.sessionManagement.sessionTimeout > 1440) {
      errors.push('Session timeout must be between 5 and 1440 minutes');
    }
  }

  if (settings.passwordPolicy) {
    if (settings.passwordPolicy.minLength < 6 || settings.passwordPolicy.minLength > 32) {
      errors.push('Password min length must be between 6 and 32');
    }
  }

  if (settings.apiSecurity?.rateLimiting) {
    if (settings.apiSecurity.rateLimiting.maxRequests < 10 || settings.apiSecurity.rateLimiting.maxRequests > 10000) {
      errors.push('Rate limit must be between 10 and 10000 requests/min');
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * GET - Retrieve security settings
 */
export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const clientIP = getClientIP(request);

    let settings = securitySettingsDB.get(sessionId);
    if (!settings) {
      settings = { ...DEFAULT_SECURITY_SETTINGS };
      settings.ipWhitelist.currentIP = clientIP;
      securitySettingsDB.set(sessionId, settings);
    }

    // Mask sensitive data
    const maskedSettings = {
      ...settings,
      twoFactorAuth: {
        ...settings.twoFactorAuth,
        secret: settings.twoFactorAuth.secret ? '***HIDDEN***' : null,
        backupCodes: settings.twoFactorAuth.backupCodes.map((code: string, i: number) =>
          i < 2 ? code : '****-****'
        ),
      },
    };

    // Calculate stats
    const stats = {
      twoFactorEnabled: settings.twoFactorAuth.enabled,
      ipWhitelistActive: settings.ipWhitelist.enabled,
      activeSessions: 1,
      totalLogins: settings.loginHistory.history.length,
      failedLogins: settings.loginHistory.history.filter((h: SecuritySettings['loginHistory']['history'][number]) => !h.success).length,
      lastLogin: settings.loginHistory.history[0]?.timestamp || null,
    };

    return NextResponse.json({
      success: true,
      data: { settings: maskedSettings, stats },
    });
  } catch (error) {
    console.error('[Security Settings API] GET Error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to get security settings' },
      { status: 500 }
    );
  }
}

/**
 * POST - Update security settings
 */
export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const clientIP = getClientIP(request);
    const body = await request.json();

    // Handle specific actions
    if (body.action === 'enable_2fa') {
      const currentSettings = securitySettingsDB.get(sessionId) || DEFAULT_SECURITY_SETTINGS;

      currentSettings.twoFactorAuth = {
        enabled: true,
        method: body.method || 'totp',
        secret: `SECRET_${Date.now()}`,
        backupCodes: generateBackupCodes(),
        lastVerified: new Date().toISOString(),
      };

      securitySettingsDB.set(sessionId, currentSettings);

      return NextResponse.json({
        success: true,
        message: '2FA enabled successfully',
        data: {
          backupCodes: currentSettings.twoFactorAuth.backupCodes,
          qrCode: 'otpauth://totp/LyTrade?secret=' + currentSettings.twoFactorAuth.secret,
        },
      });
    }

    if (body.action === 'disable_2fa') {
      const currentSettings = securitySettingsDB.get(sessionId) || DEFAULT_SECURITY_SETTINGS;

      currentSettings.twoFactorAuth = {
        enabled: false,
        method: 'totp',
        secret: null,
        backupCodes: [],
        lastVerified: null,
      };

      securitySettingsDB.set(sessionId, currentSettings);

      return NextResponse.json({
        success: true,
        message: '2FA disabled',
      });
    }

    if (body.action === 'add_ip_to_whitelist') {
      const currentSettings = securitySettingsDB.get(sessionId) || DEFAULT_SECURITY_SETTINGS;
      const ip = body.ip || clientIP;

      if (!currentSettings.ipWhitelist.allowedIPs.includes(ip)) {
        currentSettings.ipWhitelist.allowedIPs.push(ip);
        securitySettingsDB.set(sessionId, currentSettings);
      }

      return NextResponse.json({
        success: true,
        message: `IP ${ip} added to whitelist`,
        data: currentSettings,
      });
    }

    if (body.action === 'remove_ip_from_whitelist') {
      const currentSettings = securitySettingsDB.get(sessionId) || DEFAULT_SECURITY_SETTINGS;

      currentSettings.ipWhitelist.allowedIPs = currentSettings.ipWhitelist.allowedIPs.filter(
        (ip: string) => ip !== body.ip
      );

      securitySettingsDB.set(sessionId, currentSettings);

      return NextResponse.json({
        success: true,
        message: `IP ${body.ip} removed from whitelist`,
        data: currentSettings,
      });
    }

    if (body.action === 'clear_login_history') {
      const currentSettings = securitySettingsDB.get(sessionId) || DEFAULT_SECURITY_SETTINGS;
      currentSettings.loginHistory.history = [];
      securitySettingsDB.set(sessionId, currentSettings);

      return NextResponse.json({
        success: true,
        message: 'Login history cleared',
      });
    }

    // Validate input
    const validation = validateSecuritySettings(body);
    if (!validation.valid) {
      return NextResponse.json(
        { success: false, error: 'Invalid security settings', details: validation.errors },
        { status: 400 }
      );
    }

    // Get current settings or defaults
    const currentSettings = securitySettingsDB.get(sessionId) || DEFAULT_SECURITY_SETTINGS;

    // Deep merge with new settings
    const updatedSettings: SecuritySettings = {
      twoFactorAuth: { ...currentSettings.twoFactorAuth, ...(body.twoFactorAuth || {}) },
      ipWhitelist: {
        ...currentSettings.ipWhitelist,
        ...(body.ipWhitelist || {}),
        currentIP: clientIP,
      },
      sessionManagement: { ...currentSettings.sessionManagement, ...(body.sessionManagement || {}) },
      loginHistory: {
        ...currentSettings.loginHistory,
        ...(body.loginHistory || {}),
        history: body.loginHistory?.history || currentSettings.loginHistory.history,
      },
      apiSecurity: {
        ...currentSettings.apiSecurity,
        ...(body.apiSecurity || {}),
        rateLimiting: {
          ...currentSettings.apiSecurity.rateLimiting,
          ...(body.apiSecurity?.rateLimiting || {}),
        },
      },
      passwordPolicy: { ...currentSettings.passwordPolicy, ...(body.passwordPolicy || {}) },
      activityLogs: { ...currentSettings.activityLogs, ...(body.activityLogs || {}) },
    };

    securitySettingsDB.set(sessionId, updatedSettings);

    const response = NextResponse.json({
      success: true,
      message: 'Security settings updated',
      data: updatedSettings,
    });

    response.cookies.set('session_id', sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60,
    });

    return response;
  } catch (error) {
    console.error('[Security Settings API] POST Error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to update security settings' },
      { status: 500 }
    );
  }
}

/**
 * PUT - Reset to defaults
 */
export async function PUT(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    securitySettingsDB.set(sessionId, DEFAULT_SECURITY_SETTINGS);

    return NextResponse.json({
      success: true,
      message: 'Security settings reset to defaults',
      data: DEFAULT_SECURITY_SETTINGS,
    });
  } catch (error) {
    console.error('[Security Settings API] PUT Error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to reset security settings' },
      { status: 500 }
    );
  }
}
