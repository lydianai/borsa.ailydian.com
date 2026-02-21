/**
 * Coinbase Advanced Trade API Connector
 *
 * White-hat compliance: Read-only operations for balance checking and order placement
 * User provides their own API keys - platform does NOT trade on user's behalf
 *
 * Official Docs: https://docs.cdp.coinbase.com/advanced-trade/docs/welcome
 *
 * Rate Limits:
 * - Public endpoints: 10 requests per second
 * - Private endpoints: 30 requests per second
 * - Requires API Key and Secret
 *
 * CRITICAL Security:
 * - NO withdrawal permissions allowed
 * - Read + Trade permissions only
 * - All responsibility on user
 */

import crypto from 'crypto';

const COINBASE_BASE_URL = 'https://api.coinbase.com';

export interface CoinbaseCredentials {
  apiKey: string;
  apiSecret: string;
}

export interface CoinbaseAccountBalance {
  currency: string;
  available: string;
  hold: string;
  total: string;
}

export interface CoinbaseOrderRequest {
  product_id: string; // e.g., BTC-USD
  side: 'BUY' | 'SELL';
  order_configuration: {
    market_market_ioc?: { quote_size?: string; base_size?: string };
    limit_limit_gtc?: { base_size: string; limit_price: string };
  };
}

export interface CoinbaseOrderResponse {
  success: boolean;
  order_id: string;
  product_id: string;
  side: string;
}

/**
 * Generate Coinbase signature for API authentication
 * Uses CB-ACCESS-SIGN header
 */
function generateCoinbaseSignature(
  timestamp: string,
  method: string,
  requestPath: string,
  body: string,
  secret: string
): string {
  const message = timestamp + method + requestPath + body;
  return crypto
    .createHmac('sha256', secret)
    .update(message)
    .digest('base64');
}

/**
 * Make authenticated request to Coinbase API
 */
async function makeCoinbaseRequest(
  credentials: CoinbaseCredentials,
  method: 'GET' | 'POST',
  endpoint: string,
  body?: any
): Promise<any> {
  const url = COINBASE_BASE_URL + endpoint;
  const timestamp = Math.floor(Date.now() / 1000).toString();
  const bodyString = body ? JSON.stringify(body) : '';

  const signature = generateCoinbaseSignature(
    timestamp,
    method,
    endpoint,
    bodyString,
    credentials.apiSecret
  );

  const headers = {
    'CB-ACCESS-KEY': credentials.apiKey,
    'CB-ACCESS-SIGN': signature,
    'CB-ACCESS-TIMESTAMP': timestamp,
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
      throw new Error(`Coinbase API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();
    return data;
  } catch (error: any) {
    console.error('Coinbase request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Test connection and get accounts
 */
export async function testCoinbaseConnection(
  credentials: CoinbaseCredentials
): Promise<{
  success: boolean;
  accounts?: any[];
  error?: string;
}> {
  try {
    const data = await makeCoinbaseRequest(
      credentials,
      'GET',
      '/api/v3/brokerage/accounts'
    );

    if (data?.accounts) {
      return {
        success: true,
        accounts: data.accounts,
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
 * Get account balances
 */
export async function getCoinbaseBalance(
  credentials: CoinbaseCredentials
): Promise<CoinbaseAccountBalance[]> {
  try {
    const data = await makeCoinbaseRequest(
      credentials,
      'GET',
      '/api/v3/brokerage/accounts'
    );

    if (!data?.accounts) {
      return [];
    }

    return data.accounts.map((account: any) => ({
      currency: account.currency,
      available: account.available_balance?.value || '0',
      hold: account.hold?.value || '0',
      total: account.balance?.value || '0',
    }));
  } catch (error: any) {
    console.error('Failed to get Coinbase balance:', error.message);
    throw new Error('Unable to fetch balance from Coinbase');
  }
}

/**
 * Place an order
 */
export async function placeCoinbaseOrder(
  credentials: CoinbaseCredentials,
  order: CoinbaseOrderRequest
): Promise<CoinbaseOrderResponse> {
  try {
    const data = await makeCoinbaseRequest(
      credentials,
      'POST',
      '/api/v3/brokerage/orders',
      order
    );

    if (!data?.success || !data?.order_id) {
      throw new Error('Order placement failed');
    }

    return data;
  } catch (error: any) {
    console.error('Failed to place Coinbase order:', error.message);
    throw error;
  }
}

/**
 * Cancel an order
 */
export async function cancelCoinbaseOrder(
  credentials: CoinbaseCredentials,
  orderId: string
): Promise<boolean> {
  try {
    const data = await makeCoinbaseRequest(
      credentials,
      'POST',
      '/api/v3/brokerage/orders/batch_cancel',
      { order_ids: [orderId] }
    );

    return data?.results?.[0]?.success || false;
  } catch (error: any) {
    console.error('Failed to cancel Coinbase order:', error.message);
    return false;
  }
}

/**
 * Get product ticker
 */
export async function getCoinbaseTicker(
  credentials: CoinbaseCredentials,
  productId: string
): Promise<{ price: string; bid: string; ask: string } | null> {
  try {
    const data = await makeCoinbaseRequest(
      credentials,
      'GET',
      `/api/v3/brokerage/products/${productId}/ticker`
    );

    if (!data) {
      return null;
    }

    return {
      price: data.price,
      bid: data.best_bid,
      ask: data.best_ask,
    };
  } catch (error: any) {
    console.error('Failed to get Coinbase ticker:', error.message);
    return null;
  }
}
