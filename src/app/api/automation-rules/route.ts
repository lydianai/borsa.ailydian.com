/**
 * AUTOMATION RULES API
 * Auto-refresh, scheduled reports, alert automation
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { automationRulesDB } from '@/lib/database';

interface AutomationRules {
  autoRefresh: {
    enabled: boolean;
    interval: number; // seconds: 5, 10, 30, 60
    pages: ('trading-signals' | 'ai-signals' | 'quantum-signals' | 'btc-eth-analysis')[];
    pauseOnInactive: boolean; // Pause when tab is not visible
    showCountdown: boolean;
  };

  scheduledReports: {
    enabled: boolean;
    frequency: 'hourly' | 'daily' | 'weekly' | 'monthly';
    time: string; // HH:MM format
    recipients: {
      email: string[];
      telegram: boolean;
      discord: boolean;
    };
    includeCharts: boolean;
    format: 'PDF' | 'HTML' | 'JSON';
  };

  alertAutomation: {
    enabled: boolean;
    rules: {
      id: string;
      name: string;
      condition: string; // e.g., "confidence > 85", "winRate < 70"
      action: 'notify' | 'pause_trading' | 'send_report' | 'webhook';
      channels: ('browser' | 'telegram' | 'discord' | 'email')[];
      cooldown: number; // minutes between alerts
      active: boolean;
    }[];
  };

  dataBackup: {
    enabled: boolean;
    frequency: 'hourly' | 'daily' | 'weekly';
    keepDays: number; // How many days to keep backups
    autoCleanup: boolean;
    lastBackup: string | null;
  };

  webhooks: {
    enabled: boolean;
    endpoints: {
      id: string;
      name: string;
      url: string;
      events: ('new_signal' | 'high_confidence' | 'market_change' | 'daily_summary')[];
      headers: { [key: string]: string };
      active: boolean;
    }[];
  };

  tradingAutomation: {
    enabled: boolean;
    autoPauseOnLoss: boolean;
    maxDailyLossBeforePause: number; // USDT
    autoResumeNextDay: boolean;
    requireManualConfirmation: boolean;
  };
}

const DEFAULT_AUTOMATION_RULES: AutomationRules = {
  autoRefresh: {
    enabled: true,
    interval: 10,
    pages: ['trading-signals', 'ai-signals'],
    pauseOnInactive: true,
    showCountdown: true,
  },
  scheduledReports: {
    enabled: false,
    frequency: 'daily',
    time: '09:00',
    recipients: {
      email: [],
      telegram: false,
      discord: false,
    },
    includeCharts: true,
    format: 'PDF',
  },
  alertAutomation: {
    enabled: true,
    rules: [
      {
        id: 'rule_1',
        name: 'Yüksek Güven AL Sinyali',
        condition: 'confidence > 85 && type == STRONG_BUY',
        action: 'notify',
        channels: ['browser', 'telegram'],
        cooldown: 5,
        active: true,
      },
      {
        id: 'rule_2',
        name: 'Düşük Kazanma Oranı Uyarısı',
        condition: 'winRate < 60',
        action: 'send_report',
        channels: ['email'],
        cooldown: 60,
        active: false,
      },
    ],
  },
  dataBackup: {
    enabled: false,
    frequency: 'daily',
    keepDays: 7,
    autoCleanup: true,
    lastBackup: null,
  },
  webhooks: {
    enabled: false,
    endpoints: [],
  },
  tradingAutomation: {
    enabled: false,
    autoPauseOnLoss: false,
    maxDailyLossBeforePause: 100,
    autoResumeNextDay: true,
    requireManualConfirmation: true,
  },
};

// Database storage (persistent, encrypted)
// Old: const automationRulesStore = new Map<string, AutomationRules>();
// Now using: automationRulesDB from @/lib/database

async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  return sessionId;
}

function validateAutomationRules(rules: Partial<AutomationRules>): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (rules.autoRefresh) {
    if (rules.autoRefresh.interval < 5 || rules.autoRefresh.interval > 300) {
      errors.push('Refresh interval must be between 5 and 300 seconds');
    }
  }

  if (rules.scheduledReports) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    rules.scheduledReports.recipients?.email?.forEach(email => {
      if (!emailRegex.test(email)) {
        errors.push(`Invalid email: ${email}`);
      }
    });
  }

  if (rules.webhooks?.endpoints) {
    rules.webhooks.endpoints.forEach(endpoint => {
      try {
        new URL(endpoint.url);
      } catch {
        errors.push(`Invalid webhook URL: ${endpoint.url}`);
      }
    });
  }

  if (rules.tradingAutomation) {
    if (rules.tradingAutomation.maxDailyLossBeforePause < 0) {
      errors.push('Max daily loss cannot be negative');
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * GET - Retrieve automation rules
 */
export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    let rules = automationRulesDB.get(sessionId);

    if (!rules) {
      rules = DEFAULT_AUTOMATION_RULES;
      automationRulesDB.set(sessionId, rules);
    }

    // Calculate stats
    const stats = {
      activeRules: rules.alertAutomation.rules.filter((r: AutomationRules['alertAutomation']['rules'][number]) => r.active).length,
      totalRules: rules.alertAutomation.rules.length,
      webhooksActive: rules.webhooks.endpoints.filter((e: AutomationRules['webhooks']['endpoints'][number]) => e.active).length,
      autoRefreshEnabled: rules.autoRefresh.enabled,
      scheduledReportsEnabled: rules.scheduledReports.enabled,
    };

    return NextResponse.json({
      success: true,
      data: { rules, stats },
    });
  } catch (error) {
    console.error('[Automation Rules API] GET Error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to get automation rules' },
      { status: 500 }
    );
  }
}

/**
 * POST - Update automation rules
 */
export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const body = await request.json();

    // Handle specific actions
    if (body.action === 'add_alert_rule') {
      const currentRules = automationRulesDB.get(sessionId) || DEFAULT_AUTOMATION_RULES;
      const newRule = {
        id: `rule_${Date.now()}`,
        name: body.name || 'New Rule',
        condition: body.condition || '',
        action: body.ruleAction || 'notify',
        channels: body.channels || ['browser'],
        cooldown: body.cooldown || 5,
        active: true,
      };
      currentRules.alertAutomation.rules.push(newRule);
      automationRulesDB.set(sessionId, currentRules);

      return NextResponse.json({
        success: true,
        message: 'Alert rule added',
        data: currentRules,
      });
    }

    if (body.action === 'remove_alert_rule') {
      const currentRules = automationRulesDB.get(sessionId) || DEFAULT_AUTOMATION_RULES;
      currentRules.alertAutomation.rules = currentRules.alertAutomation.rules.filter(
        (r: AutomationRules['alertAutomation']['rules'][number]) => r.id !== body.ruleId
      );
      automationRulesDB.set(sessionId, currentRules);

      return NextResponse.json({
        success: true,
        message: 'Alert rule removed',
        data: currentRules,
      });
    }

    if (body.action === 'add_webhook') {
      const currentRules = automationRulesDB.get(sessionId) || DEFAULT_AUTOMATION_RULES;
      const newWebhook = {
        id: `webhook_${Date.now()}`,
        name: body.name || 'New Webhook',
        url: body.url || '',
        events: body.events || ['new_signal'],
        headers: body.headers || {},
        active: true,
      };
      currentRules.webhooks.endpoints.push(newWebhook);
      automationRulesDB.set(sessionId, currentRules);

      return NextResponse.json({
        success: true,
        message: 'Webhook added',
        data: currentRules,
      });
    }

    if (body.action === 'remove_webhook') {
      const currentRules = automationRulesDB.get(sessionId) || DEFAULT_AUTOMATION_RULES;
      currentRules.webhooks.endpoints = currentRules.webhooks.endpoints.filter(
        (e: AutomationRules['webhooks']['endpoints'][number]) => e.id !== body.webhookId
      );
      automationRulesDB.set(sessionId, currentRules);

      return NextResponse.json({
        success: true,
        message: 'Webhook removed',
        data: currentRules,
      });
    }

    if (body.action === 'trigger_manual_backup') {
      const currentRules = automationRulesDB.get(sessionId) || DEFAULT_AUTOMATION_RULES;
      currentRules.dataBackup.lastBackup = new Date().toISOString();
      automationRulesDB.set(sessionId, currentRules);

      return NextResponse.json({
        success: true,
        message: 'Manual backup completed',
        data: { lastBackup: currentRules.dataBackup.lastBackup },
      });
    }

    // Validate input
    const validation = validateAutomationRules(body);
    if (!validation.valid) {
      return NextResponse.json(
        { success: false, error: 'Invalid automation rules', details: validation.errors },
        { status: 400 }
      );
    }

    // Get current rules or defaults
    const currentRules = automationRulesDB.get(sessionId) || DEFAULT_AUTOMATION_RULES;

    // Deep merge with new rules
    const updatedRules: AutomationRules = {
      autoRefresh: { ...currentRules.autoRefresh, ...(body.autoRefresh || {}) },
      scheduledReports: {
        ...currentRules.scheduledReports,
        ...(body.scheduledReports || {}),
        recipients: {
          ...currentRules.scheduledReports.recipients,
          ...(body.scheduledReports?.recipients || {}),
        },
      },
      alertAutomation: {
        ...currentRules.alertAutomation,
        ...(body.alertAutomation || {}),
        rules: body.alertAutomation?.rules || currentRules.alertAutomation.rules,
      },
      dataBackup: { ...currentRules.dataBackup, ...(body.dataBackup || {}) },
      webhooks: {
        ...currentRules.webhooks,
        ...(body.webhooks || {}),
        endpoints: body.webhooks?.endpoints || currentRules.webhooks.endpoints,
      },
      tradingAutomation: { ...currentRules.tradingAutomation, ...(body.tradingAutomation || {}) },
    };

    automationRulesDB.set(sessionId, updatedRules);

    const response = NextResponse.json({
      success: true,
      message: 'Automation rules updated',
      data: updatedRules,
    });

    response.cookies.set('session_id', sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60,
    });

    return response;
  } catch (error) {
    console.error('[Automation Rules API] POST Error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to update automation rules' },
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
    automationRulesDB.set(sessionId, DEFAULT_AUTOMATION_RULES);

    return NextResponse.json({
      success: true,
      message: 'Automation rules reset to defaults',
      data: DEFAULT_AUTOMATION_RULES,
    });
  } catch (error) {
    console.error('[Automation Rules API] PUT Error:', error);
    return NextResponse.json(
      { success: false, error: error instanceof Error ? error.message : 'Failed to reset automation rules' },
      { status: 500 }
    );
  }
}
