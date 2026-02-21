/**
 * BTCTurk Exchange Connector
 *
 * White-hat compliance: Read-only operations for balance checking and order placement
 * User provides their own API keys - platform does NOT trade on user's behalf
 *
 * Official Docs: https://docs.btcturk.com/
 *
 * Rate Limits:
 * - Public endpoints: 100 requests per minute
 * - Private endpoints: 100 requests per minute per IP
 * - Requires API Key and Secret
 *
 * CRITICAL Security:
 * - NO withdrawal permissions allowed
 * - Read + Trade permissions only
 * - All responsibility on user
 *
 * Note: BTCTurk is Turkey's leading cryptocurrency exchange
 * Supports TRY (Turkish Lira) pairs
 */

import crypto from 'crypto';

const BTCTURK_BASE_URL = 'https://api.btcturk.com';

export interface BTCTurkCredentials {
  apiKey: string;
  apiSecret: string;
}

export interface BTCTurkAccountBalance {
  asset: string;
  assetname: string;
  balance: string;
  locked: string;
  free: string;
}

export interface BTCTurkOrderRequest {
  quantity: number;
  price?: number;
  stopPrice?: number;
  newOrderClientId?: string;
  orderMethod: 'market' | 'limit' | 'stopmarket' | 'stoplimit';
  orderType: 'buy' | 'sell';
  pairSymbol: string; // e.g., BTCTRY, ETHTRY
}

export interface BTCTurkOrderResponse {
  id: number;
  datetime: number;
  type: string;
  method: string;
  price: string;
  quantity: string;
  pairSymbol: string;
  pairSymbolNormalized: string;
  newOrderClientId: string;
}

/**
 * Generate BTCTurk signature for API authentication
 */
function generateBTCTurkSignature(
  apiKey: string,
  apiSecret: string,
  nonce: string,
  _url: string,
  body: string = ''
): string {
  const message = apiKey + nonce + body;
  return Buffer.from(
    crypto
      .createHmac('sha256', Buffer.from(apiSecret, 'base64'))
      .update(message)
      .digest()
  ).toString('base64');
}

/**
 * Make authenticated request to BTCTurk API
 */
async function makeBTCTurkRequest(
  credentials: BTCTurkCredentials,
  method: 'GET' | 'POST' | 'DELETE',
  endpoint: string,
  body?: any
): Promise<any> {
  const url = BTCTURK_BASE_URL + endpoint;
  const nonce = Date.now().toString();
  const bodyString = body ? JSON.stringify(body) : '';

  const signature = generateBTCTurkSignature(
    credentials.apiKey,
    credentials.apiSecret,
    nonce,
    endpoint,
    bodyString
  );

  const headers: Record<string, string> = {
    'X-PCK': credentials.apiKey,
    'X-Stamp': nonce,
    'X-Signature': signature,
  };

  if (method === 'POST') {
    headers['Content-Type'] = 'application/json';
  }

  try {
    const response = await fetch(url, {
      method,
      headers,
      body: body ? bodyString : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`BTCTurk API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    // BTCTurk returns { success: true/false, message, data }
    if (!data.success) {
      throw new Error(`BTCTurk error: ${data.message || 'Unknown error'}`);
    }

    return data.data;
  } catch (error: any) {
    console.error('BTCTurk request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Make public request to BTCTurk API (no auth needed)
 */
async function makeBTCTurkPublicRequest(
  endpoint: string,
  params?: Record<string, any>
): Promise<any> {
  const queryString = params ? '?' + new URLSearchParams(params).toString() : '';
  const url = `${BTCTURK_BASE_URL}${endpoint}${queryString}`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`BTCTurk public API error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(`BTCTurk error: ${data.message || 'Unknown error'}`);
    }

    return data.data;
  } catch (error: any) {
    console.error('BTCTurk public request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Test connection and get user info
 */
export async function testBTCTurkConnection(
  credentials: BTCTurkCredentials
): Promise<{
  success: boolean;
  balances?: any[];
  error?: string;
}> {
  try {
    const balances = await makeBTCTurkRequest(
      credentials,
      'GET',
      '/api/v1/users/balances'
    );

    return {
      success: true,
      balances,
    };
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
export async function getBTCTurkBalance(
  credentials: BTCTurkCredentials
): Promise<BTCTurkAccountBalance[]> {
  try {
    const balances = await makeBTCTurkRequest(
      credentials,
      'GET',
      '/api/v1/users/balances'
    );

    if (!balances) {
      return [];
    }

    return balances.map((balance: any) => ({
      asset: balance.asset,
      assetname: balance.assetname,
      balance: balance.balance,
      locked: balance.locked,
      free: balance.free,
    }));
  } catch (error: any) {
    console.error('Failed to get BTCTurk balance:', error.message);
    throw new Error('Unable to fetch balance from BTCTurk');
  }
}

/**
 * Get open orders
 */
export async function getBTCTurkOpenOrders(
  credentials: BTCTurkCredentials,
  pairSymbol?: string
): Promise<any[]> {
  try {
    const endpoint = pairSymbol
      ? `/api/v1/openOrders?pairSymbol=${pairSymbol}`
      : '/api/v1/openOrders';

    const orders = await makeBTCTurkRequest(credentials, 'GET', endpoint);

    return orders?.asks || orders?.bids || [];
  } catch (error: any) {
    console.error('Failed to get BTCTurk open orders:', error.message);
    throw new Error('Unable to fetch open orders from BTCTurk');
  }
}

/**
 * Get all orders (open + closed)
 */
export async function getBTCTurkAllOrders(
  credentials: BTCTurkCredentials,
  params?: {
    pairSymbol?: string;
    startDate?: number;
    endDate?: number;
    page?: number;
    limit?: number;
  }
): Promise<any[]> {
  try {
    const queryParams = new URLSearchParams();
    if (params?.pairSymbol) queryParams.append('pairSymbol', params.pairSymbol);
    if (params?.startDate) queryParams.append('startDate', params.startDate.toString());
    if (params?.endDate) queryParams.append('endDate', params.endDate.toString());
    if (params?.page) queryParams.append('page', params.page.toString());
    if (params?.limit) queryParams.append('limit', params.limit.toString());

    const endpoint = `/api/v1/allOrders${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    const orders = await makeBTCTurkRequest(credentials, 'GET', endpoint);

    return orders || [];
  } catch (error: any) {
    console.error('Failed to get BTCTurk all orders:', error.message);
    throw new Error('Unable to fetch orders from BTCTurk');
  }
}

/**
 * Place an order
 */
export async function placeBTCTurkOrder(
  credentials: BTCTurkCredentials,
  order: BTCTurkOrderRequest
): Promise<BTCTurkOrderResponse> {
  try {
    const data = await makeBTCTurkRequest(
      credentials,
      'POST',
      '/api/v1/order',
      order
    );

    if (!data) {
      throw new Error('Order placement failed');
    }

    return data;
  } catch (error: any) {
    console.error('Failed to place BTCTurk order:', error.message);
    throw error;
  }
}

/**
 * Cancel an order
 */
export async function cancelBTCTurkOrder(
  credentials: BTCTurkCredentials,
  orderId: number
): Promise<boolean> {
  try {
    const data = await makeBTCTurkRequest(
      credentials,
      'DELETE',
      `/api/v1/order?id=${orderId}`
    );

    return !!data;
  } catch (error: any) {
    console.error('Failed to cancel BTCTurk order:', error.message);
    return false;
  }
}

/**
 * Get ticker information (public endpoint)
 */
export async function getBTCTurkTicker(
  pairSymbol?: string
): Promise<any> {
  try {
    const endpoint = pairSymbol
      ? `/api/v2/ticker?pairSymbol=${pairSymbol}`
      : '/api/v2/ticker';

    const data = await makeBTCTurkPublicRequest(endpoint);

    return data;
  } catch (error: any) {
    console.error('Failed to get BTCTurk ticker:', error.message);
    return null;
  }
}

/**
 * Get order book (public endpoint)
 */
export async function getBTCTurkOrderBook(
  pairSymbol: string,
  limit: number = 100
): Promise<{ asks: any[]; bids: any[] } | null> {
  try {
    const data = await makeBTCTurkPublicRequest(
      `/api/v2/orderbook?pairSymbol=${pairSymbol}&limit=${limit}`
    );

    return data;
  } catch (error: any) {
    console.error('Failed to get BTCTurk order book:', error.message);
    return null;
  }
}

/**
 * Get trading pairs (public endpoint)
 */
export async function getBTCTurkTradingPairs(): Promise<any[]> {
  try {
    const data = await makeBTCTurkPublicRequest('/api/v2/server/exchangeinfo');
    return data?.symbols || [];
  } catch (error: any) {
    console.error('Failed to get BTCTurk trading pairs:', error.message);
    return [];
  }
}
