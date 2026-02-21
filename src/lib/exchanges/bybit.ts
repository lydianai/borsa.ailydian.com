/**
 * Bybit Exchange Connector
 *
 * White-hat compliance: Read-only operations for balance checking and order placement
 * User provides their own API keys - platform does NOT trade on user's behalf
 *
 * Official Docs: https://bybit-exchange.github.io/docs/v5/intro
 *
 * Rate Limits:
 * - REST API V5: 120 requests per minute (varies by endpoint)
 * - Recommended IP whitelist
 * - Requires API key and Secret (no passphrase)
 *
 * CRITICAL Security:
 * - NO withdrawal permissions allowed
 * - Read + Trade permissions only
 * - All responsibility on user
 */

import crypto from 'crypto';

const BYBIT_BASE_URL = 'https://api.bybit.com';
const BYBIT_TESTNET_URL = 'https://api-testnet.bybit.com';

export interface BybitCredentials {
  apiKey: string;
  apiSecret: string;
  testnet?: boolean;
}

export interface BybitAccountBalance {
  coin: string;
  walletBalance: string;
  availableBalance: string;
  totalPositionIM: string; // Initial margin
}

export interface BybitOrderRequest {
  category: 'spot' | 'linear' | 'inverse'; // Market type
  symbol: string; // e.g., BTCUSDT
  side: 'Buy' | 'Sell';
  orderType: 'Market' | 'Limit';
  qty: string;
  price?: string;
  timeInForce?: 'GTC' | 'IOC' | 'FOK';
  positionIdx?: number; // 0: one-way, 1: hedge-buy, 2: hedge-sell
}

export interface BybitOrderResponse {
  orderId: string;
  orderLinkId: string;
}

export interface BybitPosition {
  symbol: string;
  side: 'Buy' | 'Sell' | 'None';
  size: string;
  positionValue: string;
  entryPrice: string;
  leverage: string;
  unrealisedPnl: string;
  cumRealisedPnl: string;
  liqPrice: string;
}

/**
 * Generate Bybit signature for API authentication
 */
function generateBybitSignature(
  timestamp: string,
  apiKey: string,
  recvWindow: string,
  queryString: string,
  secret: string
): string {
  const param_str = timestamp + apiKey + recvWindow + queryString;
  return crypto
    .createHmac('sha256', secret)
    .update(param_str)
    .digest('hex');
}

/**
 * Make authenticated request to Bybit API
 */
async function makeBybitRequest(
  credentials: BybitCredentials,
  method: 'GET' | 'POST',
  endpoint: string,
  params?: Record<string, any>
): Promise<any> {
  const baseUrl = credentials.testnet ? BYBIT_TESTNET_URL : BYBIT_BASE_URL;
  const timestamp = Date.now().toString();
  const recvWindow = '5000'; // 5 seconds

  let queryString = '';
  let url = baseUrl + endpoint;

  if (method === 'GET' && params) {
    queryString = new URLSearchParams(params).toString();
    if (queryString) {
      url += '?' + queryString;
    }
  } else if (method === 'POST' && params) {
    queryString = JSON.stringify(params);
  }

  const signature = generateBybitSignature(
    timestamp,
    credentials.apiKey,
    recvWindow,
    queryString,
    credentials.apiSecret
  );

  const headers: Record<string, string> = {
    'X-BAPI-API-KEY': credentials.apiKey,
    'X-BAPI-TIMESTAMP': timestamp,
    'X-BAPI-SIGN': signature,
    'X-BAPI-RECV-WINDOW': recvWindow,
  };

  if (method === 'POST') {
    headers['Content-Type'] = 'application/json';
  }

  try {
    const response = await fetch(url, {
      method,
      headers,
      body: method === 'POST' && params ? JSON.stringify(params) : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Bybit API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    // Bybit V5 returns { retCode, retMsg, result, time }
    if (data.retCode !== 0) {
      throw new Error(`Bybit error: ${data.retMsg} (code: ${data.retCode})`);
    }

    return data.result;
  } catch (error: any) {
    console.error('Bybit request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Test connection and get API key info
 * Used to validate API keys on setup
 */
export async function testBybitConnection(
  credentials: BybitCredentials
): Promise<{
  success: boolean;
  userId?: string;
  apiKeyInfo?: any;
  error?: string;
}> {
  try {
    // Get API key information
    const data = await makeBybitRequest(
      credentials,
      'GET',
      '/v5/user/query-api'
    );

    if (data) {
      return {
        success: true,
        userId: data.id,
        apiKeyInfo: {
          readOnly: data.readOnly,
          permissions: data.permissions,
        },
      };
    }

    return { success: false, error: 'No API key info returned' };
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
export async function getBybitBalance(
  credentials: BybitCredentials,
  accountType: 'UNIFIED' | 'CONTRACT' | 'SPOT' = 'UNIFIED'
): Promise<BybitAccountBalance[]> {
  try {
    const data = await makeBybitRequest(
      credentials,
      'GET',
      '/v5/account/wallet-balance',
      { accountType }
    );

    if (!data || !data.list || data.list.length === 0) {
      return [];
    }

    const balances: BybitAccountBalance[] = [];
    const account = data.list[0];

    if (account.coin) {
      for (const coinData of account.coin) {
        balances.push({
          coin: coinData.coin,
          walletBalance: coinData.walletBalance,
          availableBalance: coinData.availableToWithdraw,
          totalPositionIM: coinData.totalPositionIM || '0',
        });
      }
    }

    return balances;
  } catch (error: any) {
    console.error('Failed to get Bybit balance:', error.message);
    throw new Error('Unable to fetch balance from Bybit');
  }
}

/**
 * Check API key permissions
 * CRITICAL: Must verify NO withdrawal permissions
 */
export async function checkBybitPermissions(
  credentials: BybitCredentials
): Promise<{
  canRead: boolean;
  canTrade: boolean;
  canWithdraw: boolean;
  permissions: string[];
}> {
  try {
    const data = await makeBybitRequest(
      credentials,
      'GET',
      '/v5/user/query-api'
    );

    if (!data) {
      return {
        canRead: false,
        canTrade: false,
        canWithdraw: false,
        permissions: [],
      };
    }

    // Bybit permissions object structure
    const permissions = data.permissions || {};
    const perms: string[] = [];

    if (permissions.ContractTrade?.includes('Order')) perms.push('trade');
    if (permissions.Spot?.includes('SpotTrade')) perms.push('spot');
    if (permissions.Wallet?.includes('AccountTransfer')) perms.push('transfer');
    if (permissions.Wallet?.includes('Withdraw')) perms.push('withdraw');

    return {
      canRead: true, // If API key works
      canTrade: perms.includes('trade') || perms.includes('spot'),
      canWithdraw: perms.includes('withdraw'),
      permissions: perms,
    };
  } catch (error: any) {
    console.error('Failed to check Bybit permissions:', error.message);
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
export async function getBybitPositions(
  credentials: BybitCredentials,
  category: 'linear' | 'inverse' = 'linear',
  symbol?: string
): Promise<BybitPosition[]> {
  try {
    const params: Record<string, any> = { category };
    if (symbol) {
      params.symbol = symbol;
    }

    const data = await makeBybitRequest(
      credentials,
      'GET',
      '/v5/position/list',
      params
    );

    return data?.list || [];
  } catch (error: any) {
    console.error('Failed to get Bybit positions:', error.message);
    throw new Error('Unable to fetch positions from Bybit');
  }
}

/**
 * Place an order
 * White-hat compliance: User-initiated orders only
 *
 * CRITICAL: Only called when user has explicitly configured auto-trading
 * with their own risk parameters
 */
export async function placeBybitOrder(
  credentials: BybitCredentials,
  order: BybitOrderRequest
): Promise<BybitOrderResponse> {
  try {
    const data = await makeBybitRequest(
      credentials,
      'POST',
      '/v5/order/create',
      order
    );

    if (!data) {
      throw new Error('No order response from Bybit');
    }

    return {
      orderId: data.orderId,
      orderLinkId: data.orderLinkId,
    };
  } catch (error: any) {
    console.error('Failed to place Bybit order:', error.message);
    throw error;
  }
}

/**
 * Cancel an order
 */
export async function cancelBybitOrder(
  credentials: BybitCredentials,
  category: 'spot' | 'linear' | 'inverse',
  symbol: string,
  orderId: string
): Promise<boolean> {
  try {
    const data = await makeBybitRequest(
      credentials,
      'POST',
      '/v5/order/cancel',
      {
        category,
        symbol,
        orderId,
      }
    );

    return !!data?.orderId;
  } catch (error: any) {
    console.error('Failed to cancel Bybit order:', error.message);
    return false;
  }
}

/**
 * Get ticker price
 */
export async function getBybitTicker(
  credentials: BybitCredentials,
  category: 'spot' | 'linear' | 'inverse',
  symbol: string
): Promise<{ lastPrice: string; bid1Price: string; ask1Price: string } | null> {
  try {
    const data = await makeBybitRequest(
      credentials,
      'GET',
      '/v5/market/tickers',
      { category, symbol }
    );

    if (!data?.list || data.list.length === 0) {
      return null;
    }

    const ticker = data.list[0];
    return {
      lastPrice: ticker.lastPrice,
      bid1Price: ticker.bid1Price,
      ask1Price: ticker.ask1Price,
    };
  } catch (error: any) {
    console.error('Failed to get Bybit ticker:', error.message);
    return null;
  }
}

/**
 * Set leverage for a symbol
 * CRITICAL: Only called when user explicitly sets leverage in bot config
 */
export async function setBybitLeverage(
  credentials: BybitCredentials,
  category: 'linear' | 'inverse',
  symbol: string,
  buyLeverage: string,
  sellLeverage: string
): Promise<boolean> {
  try {
    const data = await makeBybitRequest(
      credentials,
      'POST',
      '/v5/position/set-leverage',
      {
        category,
        symbol,
        buyLeverage,
        sellLeverage,
      }
    );

    return !!data;
  } catch (error: any) {
    console.error('Failed to set Bybit leverage:', error.message);
    throw error;
  }
}

/**
 * Set trading stop (stop loss / take profit)
 */
export async function setBybitTradingStop(
  credentials: BybitCredentials,
  category: 'linear' | 'inverse',
  symbol: string,
  stopLoss?: string,
  takeProfit?: string,
  positionIdx: number = 0
): Promise<boolean> {
  try {
    const params: Record<string, any> = {
      category,
      symbol,
      positionIdx,
    };

    if (stopLoss) params.stopLoss = stopLoss;
    if (takeProfit) params.takeProfit = takeProfit;

    const data = await makeBybitRequest(
      credentials,
      'POST',
      '/v5/position/trading-stop',
      params
    );

    return !!data;
  } catch (error: any) {
    console.error('Failed to set Bybit trading stop:', error.message);
    throw error;
  }
}
