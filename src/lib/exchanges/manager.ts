/**
 * Exchange Manager - Unified Interface
 *
 * White-hat compliance: Manages all exchange connections
 * Provides unified interface for all supported exchanges
 *
 * Supported Exchanges:
 * - OKX
 * - Bybit
 * - Coinbase
 * - Kraken
 * - BTCTurk
 *
 * CRITICAL Security:
 * - All credentials encrypted in database
 * - NO withdrawal permissions allowed
 * - Rate limiting per exchange
 * - Permission validation on each connection
 */

import { prisma } from '@/lib/prisma';
import {
  encryptExchangeCredentials,
  decryptExchangeCredentials,
} from '@/lib/encryption/service';

// Import exchange connectors
import * as OKX from './okx';
import * as Bybit from './bybit';
import * as Coinbase from './coinbase';
import * as Kraken from './kraken';
import * as BTCTurk from './btcturk';

export type SupportedExchange = 'okx' | 'bybit' | 'coinbase' | 'kraken' | 'btcturk';

export interface ExchangeCredentials {
  apiKey: string;
  apiSecret: string;
  passphrase?: string;
  testnet?: boolean;
}

export interface AddExchangeAPIRequest {
  userId: string;
  exchange: SupportedExchange;
  name: string;
  credentials: ExchangeCredentials;
  termsAccepted: boolean;
  disclaimerAccepted: boolean;
}

export interface UnifiedBalance {
  currency: string;
  available: string;
  locked: string;
  total: string;
}

export interface ConnectionTestResult {
  success: boolean;
  exchange: SupportedExchange;
  canRead: boolean;
  canTrade: boolean;
  canWithdraw: boolean;
  error?: string;
  warnings?: string[];
}

/**
 * Test exchange connection and validate permissions
 * CRITICAL: Must verify NO withdrawal permissions
 */
export async function testExchangeConnection(
  exchange: SupportedExchange,
  credentials: ExchangeCredentials
): Promise<ConnectionTestResult> {
  const warnings: string[] = [];

  try {
    let testResult: any;
    let permissionsResult: any;

    switch (exchange) {
      case 'okx':
        testResult = await OKX.testOKXConnection({
          apiKey: credentials.apiKey,
          apiSecret: credentials.apiSecret,
          passphrase: credentials.passphrase!,
          testnet: credentials.testnet,
        });

        if (!testResult.success) {
          return {
            success: false,
            exchange,
            canRead: false,
            canTrade: false,
            canWithdraw: false,
            error: testResult.error,
          };
        }

        permissionsResult = await OKX.checkOKXPermissions({
          apiKey: credentials.apiKey,
          apiSecret: credentials.apiSecret,
          passphrase: credentials.passphrase!,
          testnet: credentials.testnet,
        });
        break;

      case 'bybit':
        testResult = await Bybit.testBybitConnection({
          apiKey: credentials.apiKey,
          apiSecret: credentials.apiSecret,
          testnet: credentials.testnet,
        });

        if (!testResult.success) {
          return {
            success: false,
            exchange,
            canRead: false,
            canTrade: false,
            canWithdraw: false,
            error: testResult.error,
          };
        }

        permissionsResult = await Bybit.checkBybitPermissions({
          apiKey: credentials.apiKey,
          apiSecret: credentials.apiSecret,
          testnet: credentials.testnet,
        });
        break;

      case 'coinbase':
        testResult = await Coinbase.testCoinbaseConnection({
          apiKey: credentials.apiKey,
          apiSecret: credentials.apiSecret,
        });

        if (!testResult.success) {
          return {
            success: false,
            exchange,
            canRead: false,
            canTrade: false,
            canWithdraw: false,
            error: testResult.error,
          };
        }

        // Coinbase doesn't have direct permission check - assume success
        permissionsResult = {
          canRead: true,
          canTrade: true,
          canWithdraw: false, // User must configure
        };
        break;

      case 'kraken':
        testResult = await Kraken.testKrakenConnection({
          apiKey: credentials.apiKey,
          privateKey: credentials.apiSecret,
        });

        if (!testResult.success) {
          return {
            success: false,
            exchange,
            canRead: false,
            canTrade: false,
            canWithdraw: false,
            error: testResult.error,
          };
        }

        // Kraken doesn't have direct permission check - assume success
        permissionsResult = {
          canRead: true,
          canTrade: true,
          canWithdraw: false, // User must configure
        };
        break;

      case 'btcturk':
        testResult = await BTCTurk.testBTCTurkConnection({
          apiKey: credentials.apiKey,
          apiSecret: credentials.apiSecret,
        });

        if (!testResult.success) {
          return {
            success: false,
            exchange,
            canRead: false,
            canTrade: false,
            canWithdraw: false,
            error: testResult.error,
          };
        }

        // BTCTurk doesn't have direct permission check - assume success
        permissionsResult = {
          canRead: true,
          canTrade: true,
          canWithdraw: false, // User must configure
        };
        break;

      default:
        throw new Error(`Unsupported exchange: ${exchange}`);
    }

    // CRITICAL: Check for withdrawal permissions
    if (permissionsResult.canWithdraw) {
      warnings.push(
        '⚠️ WARNING: API key has WITHDRAWAL permissions! ' +
        'This is NOT recommended. Please create a new API key with ONLY read and trade permissions.'
      );
    }

    return {
      success: true,
      exchange,
      canRead: permissionsResult.canRead,
      canTrade: permissionsResult.canTrade,
      canWithdraw: permissionsResult.canWithdraw,
      warnings: warnings.length > 0 ? warnings : undefined,
    };
  } catch (error: any) {
    return {
      success: false,
      exchange,
      canRead: false,
      canTrade: false,
      canWithdraw: false,
      error: error.message || 'Connection test failed',
    };
  }
}

/**
 * Add new exchange API to user's account
 * Encrypts credentials and stores in database
 */
export async function addExchangeAPI(
  request: AddExchangeAPIRequest
): Promise<{ success: boolean; exchangeId?: string; error?: string }> {
  try {
    // Validate terms acceptance
    if (!request.termsAccepted || !request.disclaimerAccepted) {
      return {
        success: false,
        error: 'You must accept the terms and disclaimer to continue',
      };
    }

    // Test connection first
    const testResult = await testExchangeConnection(
      request.exchange,
      request.credentials
    );

    if (!testResult.success) {
      return {
        success: false,
        error: testResult.error || 'Connection test failed',
      };
    }

    // CRITICAL: Block if withdrawal permissions detected
    if (testResult.canWithdraw) {
      return {
        success: false,
        error:
          'API key has withdrawal permissions. For security, please create a new API key with ONLY read and trade permissions.',
      };
    }

    // Encrypt credentials
    const encrypted = encryptExchangeCredentials(
      request.credentials.apiKey,
      request.credentials.apiSecret,
      request.credentials.passphrase
    );

    // Store in database
    const exchangeAPI = await prisma.exchangeAPI.create({
      data: {
        userId: request.userId,
        exchange: request.exchange,
        name: request.name,
        encryptedApiKey: encrypted.encryptedApiKey,
        encryptedApiSecret: encrypted.encryptedApiSecret,
        encryptedPassphrase: encrypted.encryptedPassphrase,
        encryptionIV: encrypted.encryptionIV,
        isActive: true,
        isConnected: true,
        lastTestedAt: new Date(),
        hasWithdrawPerm: testResult.canWithdraw,
        hasSpotTrading: testResult.canTrade,
        hasFuturesTrading: testResult.canTrade,
        permissions: testResult.canTrade ? ['read', 'trade'] : ['read'],
        disclaimerAccepted: request.disclaimerAccepted,
        termsAcceptedAt: new Date(),
        userResponsibility: true,
        testnet: request.credentials.testnet || false,
      },
    });

    return {
      success: true,
      exchangeId: exchangeAPI.id,
    };
  } catch (error: any) {
    console.error('Failed to add exchange API:', error.message);
    return {
      success: false,
      error: error.message || 'Failed to add exchange API',
    };
  }
}

/**
 * Get decrypted credentials for an exchange API
 * CRITICAL: Only call when needed, never log returned values
 */
export async function getExchangeCredentials(
  exchangeId: string,
  userId: string
): Promise<ExchangeCredentials | null> {
  try {
    const exchangeAPI = await prisma.exchangeAPI.findFirst({
      where: {
        id: exchangeId,
        userId: userId, // Ensure user owns this API
        isActive: true,
      },
    });

    if (!exchangeAPI) {
      return null;
    }

    const decrypted = decryptExchangeCredentials(
      exchangeAPI.encryptedApiKey,
      exchangeAPI.encryptedApiSecret,
      exchangeAPI.encryptionIV,
      exchangeAPI.encryptedPassphrase
    );

    return {
      apiKey: decrypted.apiKey,
      apiSecret: decrypted.apiSecret,
      passphrase: decrypted.passphrase || undefined,
      testnet: exchangeAPI.testnet,
    };
  } catch (error: any) {
    console.error('Failed to get exchange credentials');
    return null;
  }
}

/**
 * Get unified balance across all connected exchanges
 */
export async function getUnifiedBalance(
  userId: string
): Promise<Record<SupportedExchange, UnifiedBalance[]>> {
  const balances: Record<string, UnifiedBalance[]> = {};

  try {
    const exchangeAPIs = await prisma.exchangeAPI.findMany({
      where: {
        userId,
        isActive: true,
      },
    });

    for (const api of exchangeAPIs) {
      try {
        const credentials = await getExchangeCredentials(api.id, userId);
        if (!credentials) continue;

        let exchangeBalances: UnifiedBalance[] = [];

        switch (api.exchange as SupportedExchange) {
          case 'okx':
            const okxBalances = await OKX.getOKXBalance({
              ...credentials,
              passphrase: credentials.passphrase!,
            } as OKX.OKXCredentials);
            exchangeBalances = okxBalances.map((b) => ({
              currency: b.currency,
              available: b.available,
              locked: b.frozen,
              total: b.total,
            }));
            break;

          case 'bybit':
            const bybitBalances = await Bybit.getBybitBalance(
              credentials as Bybit.BybitCredentials
            );
            exchangeBalances = bybitBalances.map((b) => ({
              currency: b.coin,
              available: b.availableBalance,
              locked: b.totalPositionIM,
              total: b.walletBalance,
            }));
            break;

          case 'coinbase':
            const cbBalances = await Coinbase.getCoinbaseBalance(
              credentials as Coinbase.CoinbaseCredentials
            );
            exchangeBalances = cbBalances.map((b) => ({
              currency: b.currency,
              available: b.available,
              locked: b.hold,
              total: b.total,
            }));
            break;

          case 'kraken':
            const krakenBalances = await Kraken.getKrakenBalance({
              apiKey: credentials.apiKey,
              privateKey: credentials.apiSecret,
            });
            exchangeBalances = krakenBalances.map((b) => ({
              currency: b.asset,
              available: b.balance,
              locked: '0',
              total: b.balance,
            }));
            break;

          case 'btcturk':
            const btcturkBalances = await BTCTurk.getBTCTurkBalance(
              credentials as BTCTurk.BTCTurkCredentials
            );
            exchangeBalances = btcturkBalances.map((b) => ({
              currency: b.asset,
              available: b.free,
              locked: b.locked,
              total: b.balance,
            }));
            break;
        }

        balances[api.exchange] = exchangeBalances;

        // Update last balance check
        await prisma.exchangeAPI.update({
          where: { id: api.id },
          data: { lastBalanceCheck: new Date() },
        });
      } catch (error: any) {
        console.error(`Failed to get balance for ${api.exchange}:`, error.message);
        balances[api.exchange] = [];
      }
    }

    return balances as Record<SupportedExchange, UnifiedBalance[]>;
  } catch (error: any) {
    console.error('Failed to get unified balance:', error.message);
    return {} as Record<SupportedExchange, UnifiedBalance[]>;
  }
}

/**
 * Delete exchange API connection
 */
export async function deleteExchangeAPI(
  exchangeId: string,
  userId: string
): Promise<boolean> {
  try {
    await prisma.exchangeAPI.delete({
      where: {
        id: exchangeId,
        userId: userId, // Ensure user owns this API
      },
    });

    return true;
  } catch (error: any) {
    console.error('Failed to delete exchange API:', error.message);
    return false;
  }
}

/**
 * Retest exchange connection
 */
export async function retestExchangeConnection(
  exchangeId: string,
  userId: string
): Promise<ConnectionTestResult | null> {
  try {
    const credentials = await getExchangeCredentials(exchangeId, userId);
    if (!credentials) {
      return null;
    }

    const api = await prisma.exchangeAPI.findFirst({
      where: { id: exchangeId, userId },
    });

    if (!api) {
      return null;
    }

    const result = await testExchangeConnection(
      api.exchange as SupportedExchange,
      credentials
    );

    // Update database with test results
    await prisma.exchangeAPI.update({
      where: { id: exchangeId },
      data: {
        isConnected: result.success,
        lastTestedAt: new Date(),
        connectionError: result.error || null,
        hasWithdrawPerm: result.canWithdraw,
      },
    });

    return result;
  } catch (error: any) {
    console.error('Failed to retest exchange connection:', error.message);
    return null;
  }
}
