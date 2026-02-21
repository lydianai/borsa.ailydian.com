/**
 * EMAIL SERVICE - Production-ready email notifications
 * Supports SMTP with retry logic, templates, and error handling
 */

import nodemailer from 'nodemailer';
import type { Transporter } from 'nodemailer';

interface EmailOptions {
  to: string;
  subject: string;
  html: string;
  text?: string;
}

interface TradeSignalData {
  symbol: string;
  signal: 'STRONG_BUY' | 'BUY' | 'SELL' | 'STRONG_SELL';
  price: number;
  confidence: number;
  timestamp: string;
}

class EmailService {
  private transporter: Transporter | null = null;
  private retryCount = 3;
  private retryDelay = 1000;
  private isConfigured = false;

  constructor() {
    this.initialize();
  }

  private async initialize() {
    try {
      const smtpHost = process.env.SMTP_HOST;
      const smtpPort = process.env.SMTP_PORT;
      const smtpUser = process.env.SMTP_USER;
      const smtpPass = process.env.SMTP_PASS;
      const smtpFrom = process.env.SMTP_FROM;

      if (!smtpHost || !smtpPort || !smtpUser || !smtpPass || !smtpFrom) {
        console.warn('[Email Service] SMTP not configured - email sending disabled');
        return;
      }

      this.transporter = nodemailer.createTransport({
        host: smtpHost,
        port: parseInt(smtpPort, 10),
        secure: parseInt(smtpPort, 10) === 465,
        auth: {
          user: smtpUser,
          pass: smtpPass,
        },
        pool: true,
        maxConnections: 5,
        maxMessages: 100,
        rateDelta: 1000,
        rateLimit: 10,
      });

      await this.transporter.verify();
      this.isConfigured = true;
      console.log('[Email Service] SMTP configured and verified successfully');
    } catch (error) {
      console.error('[Email Service] Initialization failed:', error);
      this.isConfigured = false;
    }
  }

  private async sendWithRetry(options: EmailOptions, attempt = 1): Promise<boolean> {
    if (!this.transporter || !this.isConfigured) {
      console.warn('[Email Service] Email not sent - service not configured');
      return false;
    }

    try {
      const from = process.env.SMTP_FROM || 'noreply@sardag-trading.com';

      await this.transporter.sendMail({
        from,
        to: options.to,
        subject: options.subject,
        html: options.html,
        text: options.text || this.htmlToText(options.html),
      });

      console.log(`[Email Service] Email sent successfully to ${options.to}`);
      return true;
    } catch (error) {
      console.error(`[Email Service] Send attempt ${attempt} failed:`, error);

      if (attempt < this.retryCount) {
        await this.sleep(this.retryDelay * attempt);
        return this.sendWithRetry(options, attempt + 1);
      }

      return false;
    }
  }

  private htmlToText(html: string): string {
    return html
      .replace(/<[^>]*>/g, '')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .trim();
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async sendTradeSignal(to: string, data: TradeSignalData): Promise<boolean> {
    const signalColor =
      data.signal.includes('BUY') ? '#00ff00' :
      data.signal.includes('SELL') ? '#ff0000' : '#ffff00';

    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trade Signal Alert</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #ffffff;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 40px auto; background: #1a1a1a; border: 1px solid #333; border-radius: 12px; overflow: hidden;">
    <tr>
      <td style="padding: 32px; text-align: center; background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); border-bottom: 1px solid ${signalColor};">
        <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff;">SARDAG Trading Signal</h1>
        <p style="margin: 8px 0 0 0; font-size: 14px; color: #888;">Automated Trading Alert</p>
      </td>
    </tr>
    <tr>
      <td style="padding: 32px;">
        <div style="text-align: center; margin-bottom: 24px;">
          <div style="display: inline-block; padding: 12px 24px; background: ${signalColor}20; border: 2px solid ${signalColor}; border-radius: 8px; margin-bottom: 16px;">
            <div style="font-size: 14px; color: ${signalColor}; font-weight: 600; letter-spacing: 1px;">SIGNAL</div>
            <div style="font-size: 32px; font-weight: 700; color: ${signalColor}; margin-top: 4px;">${data.signal.replace('_', ' ')}</div>
          </div>
        </div>

        <table width="100%" cellpadding="0" cellspacing="0" style="margin: 24px 0;">
          <tr>
            <td style="padding: 16px; background: #0a0a0a; border-radius: 8px; margin-bottom: 12px;">
              <div style="font-size: 13px; color: #888; margin-bottom: 4px;">Symbol</div>
              <div style="font-size: 24px; font-weight: 700; color: #ffffff;">${data.symbol}</div>
            </td>
          </tr>
          <tr><td style="height: 12px;"></td></tr>
          <tr>
            <td style="padding: 16px; background: #0a0a0a; border-radius: 8px;">
              <div style="font-size: 13px; color: #888; margin-bottom: 4px;">Current Price</div>
              <div style="font-size: 20px; font-weight: 600; color: #00ffff;">$${data.price.toFixed(2)}</div>
            </td>
          </tr>
          <tr><td style="height: 12px;"></td></tr>
          <tr>
            <td style="padding: 16px; background: #0a0a0a; border-radius: 8px;">
              <div style="font-size: 13px; color: #888; margin-bottom: 4px;">Confidence</div>
              <div style="font-size: 20px; font-weight: 600; color: #00ff00;">${(data.confidence * 100).toFixed(1)}%</div>
            </td>
          </tr>
        </table>

        <div style="margin-top: 24px; padding: 16px; background: #0a0a0a; border-left: 3px solid #00ffff; border-radius: 4px;">
          <div style="font-size: 13px; color: #888;">Timestamp</div>
          <div style="font-size: 14px; color: #ffffff; margin-top: 4px;">${new Date(data.timestamp).toLocaleString()}</div>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding: 24px; text-align: center; background: #0a0a0a; border-top: 1px solid #333;">
        <p style="margin: 0; font-size: 12px; color: #666;">
          This is an automated trading signal from SARDAG Trading System<br>
          Do not reply to this email
        </p>
      </td>
    </tr>
  </table>
</body>
</html>
    `;

    return this.sendWithRetry({
      to,
      subject: `${data.signal} Alert: ${data.symbol} @ $${data.price.toFixed(2)}`,
      html,
    });
  }

  async sendDailySummary(to: string, data: {
    date: string;
    totalSignals: number;
    strongBuy: number;
    buy: number;
    sell: number;
    strongSell: number;
    topPerformers: Array<{ symbol: string; gain: number }>;
  }): Promise<boolean> {
    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Daily Trading Summary</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #ffffff;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 40px auto; background: #1a1a1a; border: 1px solid #333; border-radius: 12px; overflow: hidden;">
    <tr>
      <td style="padding: 32px; text-align: center; background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); border-bottom: 1px solid #00ff00;">
        <h1 style="margin: 0; font-size: 28px; font-weight: 700; color: #ffffff;">Daily Trading Summary</h1>
        <p style="margin: 8px 0 0 0; font-size: 14px; color: #888;">${data.date}</p>
      </td>
    </tr>
    <tr>
      <td style="padding: 32px;">
        <h2 style="margin: 0 0 16px 0; font-size: 18px; font-weight: 600; color: #ffffff;">Signal Overview</h2>
        <div style="background: #0a0a0a; padding: 20px; border-radius: 8px; margin-bottom: 24px;">
          <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px;">
            <div style="text-align: center;">
              <div style="font-size: 32px; font-weight: 700; color: #00ff00;">${data.strongBuy}</div>
              <div style="font-size: 13px; color: #888; margin-top: 4px;">Strong Buy</div>
            </div>
            <div style="text-align: center;">
              <div style="font-size: 32px; font-weight: 700; color: #00cc00;">${data.buy}</div>
              <div style="font-size: 13px; color: #888; margin-top: 4px;">Buy</div>
            </div>
            <div style="text-align: center;">
              <div style="font-size: 32px; font-weight: 700; color: #ff9900;">${data.sell}</div>
              <div style="font-size: 13px; color: #888; margin-top: 4px;">Sell</div>
            </div>
            <div style="text-align: center;">
              <div style="font-size: 32px; font-weight: 700; color: #ff0000;">${data.strongSell}</div>
              <div style="font-size: 13px; color: #888; margin-top: 4px;">Strong Sell</div>
            </div>
          </div>
        </div>

        ${data.topPerformers.length > 0 ? `
        <h2 style="margin: 24px 0 16px 0; font-size: 18px; font-weight: 600; color: #ffffff;">Top Performers</h2>
        ${data.topPerformers.map(coin => `
        <div style="background: #0a0a0a; padding: 16px; border-radius: 8px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
          <div style="font-size: 16px; font-weight: 600; color: #ffffff;">${coin.symbol}</div>
          <div style="font-size: 18px; font-weight: 700; color: ${coin.gain >= 0 ? '#00ff00' : '#ff0000'};">${coin.gain >= 0 ? '+' : ''}${coin.gain.toFixed(2)}%</div>
        </div>
        `).join('')}
        ` : ''}
      </td>
    </tr>
    <tr>
      <td style="padding: 24px; text-align: center; background: #0a0a0a; border-top: 1px solid #333;">
        <p style="margin: 0; font-size: 12px; color: #666;">
          SARDAG Trading System - Daily Summary Report<br>
          Do not reply to this email
        </p>
      </td>
    </tr>
  </table>
</body>
</html>
    `;

    return this.sendWithRetry({
      to,
      subject: `SARDAG Daily Summary - ${data.date} (${data.totalSignals} signals)`,
      html,
    });
  }

  async sendAlert(to: string, title: string, message: string, type: 'info' | 'warning' | 'error' = 'info'): Promise<boolean> {
    const colors = {
      info: '#00ffff',
      warning: '#ffff00',
      error: '#ff0000',
    };

    const color = colors[type];

    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title}</title>
</head>
<body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #ffffff;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 40px auto; background: #1a1a1a; border: 1px solid #333; border-radius: 12px; overflow: hidden;">
    <tr>
      <td style="padding: 32px; text-align: center; background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); border-bottom: 1px solid ${color};">
        <h1 style="margin: 0; font-size: 24px; font-weight: 700; color: ${color};">${title}</h1>
      </td>
    </tr>
    <tr>
      <td style="padding: 32px;">
        <div style="background: #0a0a0a; padding: 24px; border-radius: 8px; border-left: 4px solid ${color};">
          <p style="margin: 0; font-size: 16px; line-height: 1.6; color: #ffffff;">${message}</p>
        </div>
      </td>
    </tr>
    <tr>
      <td style="padding: 24px; text-align: center; background: #0a0a0a; border-top: 1px solid #333;">
        <p style="margin: 0; font-size: 12px; color: #666;">
          SARDAG Trading System Alert<br>
          Do not reply to this email
        </p>
      </td>
    </tr>
  </table>
</body>
</html>
    `;

    return this.sendWithRetry({
      to,
      subject: `SARDAG Alert: ${title}`,
      html,
    });
  }

  getStatus(): { configured: boolean; ready: boolean } {
    return {
      configured: this.isConfigured,
      ready: this.transporter !== null && this.isConfigured,
    };
  }
}

export const emailService = new EmailService();
