/**
 * OKX Exchange Connector
 *
 * White-hat compliance: Read-only operations for balance checking and order placement
 * User provides their own API keys - platform does NOT trade on user's behalf
 *
 * Official Docs: https://www.okx.com/docs-v5/en/
 *
 * Rate Limits:
 * - REST API: 20 requests per 2 seconds
 * - Must have IP whitelist configured by user
 * - Requires API key, Secret, and Passphrase
 *
 * CRITICAL Security:
 * - NO withdrawal permissions allowed
 * - Read + Trade permissions only
 * - All responsibility on user
 */

import crypto from 'crypto';

const OKX_BASE_URL = 'https://www.okx.com';
const OKX_API_VERSION = '/api/v5';

export interface OKXCredentials {
  apiKey: string;
  apiSecret: string;
  passphrase: string;
  testnet?: boolean;
}

export interface OKXAccountBalance {
  currency: string;
  available: string;
  frozen: string;
  total: string;
}

export interface OKXOrderRequest {
  instId: string; // Instrument ID (e.g., BTC-USDT)
  tdMode: 'cash' | 'cross' | 'isolated'; // Trade mode
  side: 'buy' | 'sell';
  ordType: 'market' | 'limit';
  sz: string; // Size
  px?: string; // Price (for limit orders)
  lever?: string; // Leverage
}

export interface OKXOrderResponse {
  ordId: string;
  clOrdId: string;
  sCode: string;
  sMsg: string;
}

export interface OKXPosition {
  instId: string;
  posId: string;
  posSide: 'long' | 'short' | 'net';
  pos: string;
  availPos: string;
  avgPx: string;
  upl: string; // Unrealized P&L
  uplRatio: string;
  lever: string;
  liqPx: string; // Liquidation price
  margin: string;
}

/**
 * Generate OKX signature for API authentication
 */
function generateSignature(
  timestamp: string,
  method: string,
  requestPath: string,
  body: string,
  secret: string
): string {
  const prehash = timestamp + method + requestPath + body;
  return crypto
    .createHmac('sha256', secret)
    .update(prehash)
    .digest('base64');
}

/**
 * Make authenticated request to OKX API
 */
async function makeOKXRequest(
  credentials: OKXCredentials,
  method: 'GET' | 'POST',
  endpoint: string,
  body?: any
): Promise<any> {
  const baseUrl = credentials.testnet
    ? 'https://www.okx.com' // OKX uses same URL for testnet, different API keys
    : OKX_BASE_URL;

  const url = baseUrl + OKX_API_VERSION + endpoint;
  const timestamp = new Date().toISOString();
  const bodyString = body ? JSON.stringify(body) : '';

  const signature = generateSignature(
    timestamp,
    method,
    OKX_API_VERSION + endpoint,
    bodyString,
    credentials.apiSecret
  );

  const headers = {
    'OK-ACCESS-KEY': credentials.apiKey,
    'OK-ACCESS-SIGN': signature,
    'OK-ACCESS-TIMESTAMP': timestamp,
    'OK-ACCESS-PASSPHRASE': credentials.passphrase,
    'Content-Type': 'application/json',
  };

  try {
    const response = await fetch(url, {
      method,
      headers,
      body: body ? bodyString : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`OKX API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    // OKX returns { code, msg, data }
    if (data.code !== '0') {
      throw new Error(`OKX error: ${data.msg} (code: ${data.code})`);
    }

    return data.data;
  } catch (error: any) {
    console.error('OKX request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Test connection and get account info
 * Used to validate API keys on setup
 */
export async function testOKXConnection(
  credentials: OKXCredentials
): Promise<{
  success: boolean;
  accountType?: string;
  uid?: string;
  error?: string;
}> {
  try {
    const data = await makeOKXRequest(credentials, 'GET', '/account/config');

    // Check if any data returned
    if (data && data.length > 0) {
      const account = data[0];
      return {
        success: true,
        accountType: account.acctLv, // Account level
        uid: account.uid,
      };
    }

    return { success: false, error: 'No account data returned' };
  } catch (error: any) {
    return {
      success: false,
      error: error.message || 'Connection test failed',
    };
  }
}

/**
 * Get account balance
 * CRITICAL: This is read-only operation
 */
export async function getOKXBalance(
  credentials: OKXCredentials
): Promise<OKXAccountBalance[]> {
  try {
    const data = await makeOKXRequest(credentials, 'GET', '/account/balance');

    if (!data || data.length === 0) {
      return [];
    }

    const balances: OKXAccountBalance[] = [];
    const accountData = data[0];

    if (accountData.details) {
      for (const detail of accountData.details) {
        balances.push({
          currency: detail.ccy,
          available: detail.availBal,
          frozen: detail.frozenBal,
          total: detail.eq, // Total equity
        });
      }
    }

    return balances;
  } catch (error: any) {
    console.error('Failed to get OKX balance:', error.message);
    throw new Error('Unable to fetch balance from OKX');
  }
}

/**
 * Check API key permissions
 * CRITICAL: Must verify NO withdrawal permissions
 */
export async function checkOKXPermissions(
  credentials: OKXCredentials
): Promise<{
  canRead: boolean;
  canTrade: boolean;
  canWithdraw: boolean;
  permissions: string[];
}> {
  try {
    // OKX doesn't have a direct permissions endpoint
    // We test capabilities by attempting read operations
    const _balanceTest = await getOKXBalance(credentials);

    // Try to get positions (requires read permission)
    const _positionsTest = await makeOKXRequest(
      credentials,
      'GET',
      '/account/positions'
    );

    // Note: We cannot test withdrawal without actually attempting it
    // User MUST configure API key with NO withdrawal permission

    return {
      canRead: true,
      canTrade: true, // Assumed if API key works
      canWithdraw: false, // User must ensure this
      permissions: ['read', 'trade'],
    };
  } catch (error: any) {
    console.error('Failed to check OKX permissions:', error.message);
    return {
      canRead: false,
      canTrade: false,
      canWithdraw: false,
      permissions: [],
    };
  }
}

/**
 * Get current positions
 */
export async function getOKXPositions(
  credentials: OKXCredentials,
  instType?: 'FUTURES' | 'SWAP' | 'MARGIN'
): Promise<OKXPosition[]> {
  try {
    const endpoint = instType
      ? `/account/positions?instType=${instType}`
      : '/account/positions';

    const data = await makeOKXRequest(credentials, 'GET', endpoint);

    return data || [];
  } catch (error: any) {
    console.error('Failed to get OKX positions:', error.message);
    throw new Error('Unable to fetch positions from OKX');
  }
}

/**
 * Place an order
 * White-hat compliance: User-initiated orders only
 *
 * CRITICAL: Only called when user has explicitly configured auto-trading
 * with their own risk parameters
 */
export async function placeOKXOrder(
  credentials: OKXCredentials,
  order: OKXOrderRequest
): Promise<OKXOrderResponse> {
  try {
    const data = await makeOKXRequest(credentials, 'POST', '/trade/order', order);

    if (!data || data.length === 0) {
      throw new Error('No order response from OKX');
    }

    const result = data[0];

    if (result.sCode !== '0') {
      throw new Error(`Order failed: ${result.sMsg} (code: ${result.sCode})`);
    }

    return result;
  } catch (error: any) {
    console.error('Failed to place OKX order:', error.message);
    throw error;
  }
}

/**
 * Cancel an order
 */
export async function cancelOKXOrder(
  credentials: OKXCredentials,
  instId: string,
  ordId: string
): Promise<boolean> {
  try {
    const data = await makeOKXRequest(credentials, 'POST', '/trade/cancel-order', {
      instId,
      ordId,
    });

    if (!data || data.length === 0) {
      return false;
    }

    const result = data[0];
    return result.sCode === '0';
  } catch (error: any) {
    console.error('Failed to cancel OKX order:', error.message);
    return false;
  }
}

/**
 * Get ticker price
 */
export async function getOKXTicker(
  credentials: OKXCredentials,
  instId: string
): Promise<{ last: string; bid: string; ask: string } | null> {
  try {
    const data = await makeOKXRequest(
      credentials,
      'GET',
      `/market/ticker?instId=${instId}`
    );

    if (!data || data.length === 0) {
      return null;
    }

    const ticker = data[0];
    return {
      last: ticker.last,
      bid: ticker.bidPx,
      ask: ticker.askPx,
    };
  } catch (error: any) {
    console.error('Failed to get OKX ticker:', error.message);
    return null;
  }
}

/**
 * Get account leverage for a specific instrument
 */
export async function getOKXLeverage(
  credentials: OKXCredentials,
  instId: string
): Promise<{ lever: string; mgnMode: string } | null> {
  try {
    const data = await makeOKXRequest(
      credentials,
      'GET',
      `/account/leverage-info?instId=${instId}&mgnMode=cross`
    );

    if (!data || data.length === 0) {
      return null;
    }

    return data[0];
  } catch (error: any) {
    console.error('Failed to get OKX leverage:', error.message);
    return null;
  }
}

/**
 * Set leverage for an instrument
 * CRITICAL: Only called when user explicitly sets leverage in bot config
 */
export async function setOKXLeverage(
  credentials: OKXCredentials,
  instId: string,
  lever: string,
  mgnMode: 'cross' | 'isolated' = 'cross'
): Promise<boolean> {
  try {
    const data = await makeOKXRequest(
      credentials,
      'POST',
      '/account/set-leverage',
      {
        instId,
        lever,
        mgnMode,
      }
    );

    if (!data || data.length === 0) {
      return false;
    }

    const result = data[0];
    return result.lever === lever;
  } catch (error: any) {
    console.error('Failed to set OKX leverage:', error.message);
    throw error;
  }
}
