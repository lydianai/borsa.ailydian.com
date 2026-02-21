/**
 * TWO-FACTOR AUTHENTICATION SERVICE
 * Production-ready TOTP-based 2FA with QR code generation and backup codes
 */

import { authenticator } from 'otplib';
import QRCode from 'qrcode';
import { randomBytes } from 'crypto';

interface TwoFactorSecret {
  secret: string;
  qrCodeDataUrl: string;
  backupCodes: string[];
}

interface VerificationResult {
  valid: boolean;
  message: string;
}

class TwoFactorAuthService {
  private readonly appName = 'SARDAG Trading';
  private readonly window = 1;

  constructor() {
    authenticator.options = {
      window: this.window,
      step: 30,
    };
  }

  async generateSecret(userEmail: string): Promise<TwoFactorSecret> {
    try {
      const secret = authenticator.generateSecret();

      const otpauthUrl = authenticator.keyuri(
        userEmail,
        this.appName,
        secret
      );

      const qrCodeDataUrl = await QRCode.toDataURL(otpauthUrl, {
        errorCorrectionLevel: 'H',
        type: 'image/png',
        width: 300,
        margin: 2,
        color: {
          dark: '#000000',
          light: '#FFFFFF',
        },
      });

      const backupCodes = this.generateBackupCodes(10);

      console.log('[2FA Service] Secret generated successfully for:', userEmail);

      return {
        secret,
        qrCodeDataUrl,
        backupCodes,
      };
    } catch (error) {
      console.error('[2FA Service] Failed to generate secret:', error);
      throw new Error('Failed to generate 2FA secret');
    }
  }

  verifyToken(token: string, secret: string): VerificationResult {
    try {
      if (!token || !secret) {
        return {
          valid: false,
          message: 'Token and secret are required',
        };
      }

      if (!/^\d{6}$/.test(token)) {
        return {
          valid: false,
          message: 'Invalid token format. Must be 6 digits.',
        };
      }

      const isValid = authenticator.verify({
        token,
        secret,
      });

      return {
        valid: isValid,
        message: isValid ? 'Token verified successfully' : 'Invalid or expired token',
      };
    } catch (error) {
      console.error('[2FA Service] Token verification error:', error);
      return {
        valid: false,
        message: 'Token verification failed',
      };
    }
  }

  verifyBackupCode(code: string, backupCodes: string[]): { valid: boolean; remainingCodes: string[] } {
    try {
      const normalizedCode = code.replace(/\s|-/g, '').toUpperCase();

      const index = backupCodes.findIndex(bc => bc === normalizedCode);

      if (index === -1) {
        return {
          valid: false,
          remainingCodes: backupCodes,
        };
      }

      const remainingCodes = backupCodes.filter((_, i) => i !== index);

      console.log('[2FA Service] Backup code used. Remaining codes:', remainingCodes.length);

      return {
        valid: true,
        remainingCodes,
      };
    } catch (error) {
      console.error('[2FA Service] Backup code verification error:', error);
      return {
        valid: false,
        remainingCodes: backupCodes,
      };
    }
  }

  generateBackupCodes(count: number = 10): string[] {
    const codes: string[] = [];

    for (let i = 0; i < count; i++) {
      const code = randomBytes(4).toString('hex').toUpperCase();
      const formatted = `${code.substring(0, 4)}-${code.substring(4, 8)}`;
      codes.push(formatted);
    }

    return codes;
  }

  generateRecoveryCodes(count: number = 5): string[] {
    const codes: string[] = [];

    for (let i = 0; i < count; i++) {
      const code = randomBytes(8).toString('base64').replace(/[^a-zA-Z0-9]/g, '').substring(0, 16).toUpperCase();
      const formatted = `${code.substring(0, 4)}-${code.substring(4, 8)}-${code.substring(8, 12)}-${code.substring(12, 16)}`;
      codes.push(formatted);
    }

    return codes;
  }

  getCurrentToken(secret: string): string {
    try {
      return authenticator.generate(secret);
    } catch (error) {
      console.error('[2FA Service] Failed to generate current token:', error);
      return '';
    }
  }

  getRemainingTime(): number {
    const epoch = Math.round(new Date().getTime() / 1000.0);
    const countDown = 30 - (epoch % 30);
    return countDown;
  }

  validateSecret(secret: string): boolean {
    try {
      if (!secret || secret.length < 16) {
        return false;
      }

      const testToken = authenticator.generate(secret);
      return /^\d{6}$/.test(testToken);
    } catch {
      return false;
    }
  }

  getQRCodeSvg(otpauthUrl: string): Promise<string> {
    return QRCode.toString(otpauthUrl, {
      type: 'svg',
      errorCorrectionLevel: 'H',
      width: 300,
      margin: 2,
    });
  }

  async generateQRCodeBuffer(otpauthUrl: string): Promise<Buffer> {
    return QRCode.toBuffer(otpauthUrl, {
      errorCorrectionLevel: 'H',
      type: 'png',
      width: 300,
      margin: 2,
      color: {
        dark: '#000000',
        light: '#FFFFFF',
      },
    });
  }

  maskSecret(secret: string): string {
    if (secret.length <= 8) {
      return '*'.repeat(secret.length);
    }

    const visible = 4;
    const start = secret.substring(0, visible);
    const end = secret.substring(secret.length - visible);
    const masked = '*'.repeat(secret.length - visible * 2);

    return `${start}${masked}${end}`;
  }

  generateSessionToken(): string {
    return randomBytes(32).toString('hex');
  }

  hashBackupCode(code: string): string {
    const crypto = require('crypto');
    return crypto
      .createHash('sha256')
      .update(code)
      .digest('hex');
  }
}

export const twoFactorAuthService = new TwoFactorAuthService();
export type { TwoFactorSecret, VerificationResult };
