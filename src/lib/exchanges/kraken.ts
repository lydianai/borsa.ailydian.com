/**
 * Kraken Exchange Connector
 *
 * White-hat compliance: Read-only operations for balance checking and order placement
 * User provides their own API keys - platform does NOT trade on user's behalf
 *
 * Official Docs: https://docs.kraken.com/rest/
 *
 * Rate Limits:
 * - Tier-based system (varies by verification level)
 * - Typically: 15-20 requests per second for private endpoints
 * - Requires API Key and Private Key
 *
 * CRITICAL Security:
 * - NO withdrawal permissions allowed
 * - Read + Trade permissions only
 * - All responsibility on user
 */

import crypto from 'crypto';

const KRAKEN_BASE_URL = 'https://api.kraken.com';

export interface KrakenCredentials {
  apiKey: string;
  privateKey: string; // Base64 encoded private key
}

export interface KrakenAccountBalance {
  asset: string;
  balance: string;
}

export interface KrakenOrderRequest {
  pair: string; // e.g., XXBTZUSD (BTC/USD)
  type: 'buy' | 'sell';
  ordertype: 'market' | 'limit';
  volume: string;
  price?: string;
  leverage?: string;
}

export interface KrakenOrderResponse {
  txid: string[];
  descr: {
    order: string;
  };
}

/**
 * Generate Kraken signature for API authentication
 */
function generateKrakenSignature(
  path: string,
  nonce: string,
  postData: string,
  privateKey: string
): string {
  const message = path + crypto
    .createHash('sha256')
    .update(nonce + postData)
    .digest();

  const secret = Buffer.from(privateKey, 'base64');
  return crypto
    .createHmac('sha512', secret)
    .update(message)
    .digest('base64');
}

/**
 * Make authenticated request to Kraken API
 */
async function makeKrakenRequest(
  credentials: KrakenCredentials,
  endpoint: string,
  params?: Record<string, any>
): Promise<any> {
  const path = `/0/private/${endpoint}`;
  const url = KRAKEN_BASE_URL + path;
  const nonce = Date.now().toString() + '000'; // Microseconds

  const postData = new URLSearchParams({
    nonce,
    ...params,
  }).toString();

  const signature = generateKrakenSignature(
    path,
    nonce,
    postData,
    credentials.privateKey
  );

  const headers = {
    'API-Key': credentials.apiKey,
    'API-Sign': signature,
    'Content-Type': 'application/x-www-form-urlencoded',
  };

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: postData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Kraken API error: ${response.status} - ${errorText}`);
    }

    const data = await response.json();

    // Kraken returns { error: [], result: {} }
    if (data.error && data.error.length > 0) {
      throw new Error(`Kraken error: ${data.error.join(', ')}`);
    }

    return data.result;
  } catch (error: any) {
    console.error('Kraken request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Make public request to Kraken API (no auth needed)
 */
async function makeKrakenPublicRequest(
  endpoint: string,
  params?: Record<string, any>
): Promise<any> {
  const queryString = params ? '?' + new URLSearchParams(params).toString() : '';
  const url = `${KRAKEN_BASE_URL}/0/public/${endpoint}${queryString}`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Kraken public API error: ${response.status}`);
    }

    const data = await response.json();

    if (data.error && data.error.length > 0) {
      throw new Error(`Kraken error: ${data.error.join(', ')}`);
    }

    return data.result;
  } catch (error: any) {
    console.error('Kraken public request failed:', endpoint, error.message);
    throw error;
  }
}

/**
 * Test connection and get account balance
 */
export async function testKrakenConnection(
  credentials: KrakenCredentials
): Promise<{
  success: boolean;
  balances?: Record<string, string>;
  error?: string;
}> {
  try {
    const balances = await makeKrakenRequest(credentials, 'Balance');
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
export async function getKrakenBalance(
  credentials: KrakenCredentials
): Promise<KrakenAccountBalance[]> {
  try {
    const balances = await makeKrakenRequest(credentials, 'Balance');

    if (!balances) {
      return [];
    }

    return Object.entries(balances).map(([asset, balance]) => ({
      asset,
      balance: balance as string,
    }));
  } catch (error: any) {
    console.error('Failed to get Kraken balance:', error.message);
    throw new Error('Unable to fetch balance from Kraken');
  }
}

/**
 * Get open positions
 */
export async function getKrakenPositions(
  credentials: KrakenCredentials
): Promise<any[]> {
  try {
    const data = await makeKrakenRequest(credentials, 'OpenPositions');

    if (!data) {
      return [];
    }

    return Object.entries(data).map(([txid, position]) => ({
      txid,
      ...(position as object),
    }));
  } catch (error: any) {
    console.error('Failed to get Kraken positions:', error.message);
    throw new Error('Unable to fetch positions from Kraken');
  }
}

/**
 * Place an order
 */
export async function placeKrakenOrder(
  credentials: KrakenCredentials,
  order: KrakenOrderRequest
): Promise<KrakenOrderResponse> {
  try {
    const data = await makeKrakenRequest(credentials, 'AddOrder', order);

    if (!data || !data.txid) {
      throw new Error('Order placement failed');
    }

    return data;
  } catch (error: any) {
    console.error('Failed to place Kraken order:', error.message);
    throw error;
  }
}

/**
 * Cancel an order
 */
export async function cancelKrakenOrder(
  credentials: KrakenCredentials,
  txid: string
): Promise<boolean> {
  try {
    const data = await makeKrakenRequest(credentials, 'CancelOrder', { txid });
    return data?.count > 0;
  } catch (error: any) {
    console.error('Failed to cancel Kraken order:', error.message);
    return false;
  }
}

/**
 * Get ticker information (public endpoint)
 */
export async function getKrakenTicker(
  pair: string
): Promise<{ last: string; bid: string; ask: string } | null> {
  try {
    const data = await makeKrakenPublicRequest('Ticker', { pair });

    if (!data || !data[pair]) {
      return null;
    }

    const ticker = data[pair];
    return {
      last: ticker.c[0], // Last trade price
      bid: ticker.b[0], // Bid price
      ask: ticker.a[0], // Ask price
    };
  } catch (error: any) {
    console.error('Failed to get Kraken ticker:', error.message);
    return null;
  }
}
