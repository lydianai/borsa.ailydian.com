/**
 * API KEY MANAGEMENT API
 * Manages all external API keys: Binance, Groq AI, CoinMarketCap, Telegram
 *
 * Features:
 * - Secure key storage (masked in responses)
 * - Test connection for each API
 * - Validation for key formats
 * - Status tracking (connected/disconnected)
 * - Last tested timestamps
 */

import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import crypto from 'crypto';
import { apiKeysDB } from '@/lib/database';

// Obfuscated API endpoints
const _modelsUrl = Buffer.from('aHR0cHM6Ly9hcGkuZ3JvcS5jb20vb3BlbmFpL3YxL21vZGVscw==', 'base64').toString('utf-8');

// API Keys Interface
interface APIKeys {
  binance: {
    apiKey: string;
    secretKey: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
    testnet: boolean;
  };
  okx: {
    apiKey: string;
    secretKey: string;
    passphrase: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
    testnet: boolean;
  };
  bybit: {
    apiKey: string;
    secretKey: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
    testnet: boolean;
  };
  btcturk: {
    apiKey: string;
    secretKey: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
  };
  groq: {
    apiKey: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
    model: 'llama-3.3-70b-versatile' | 'llama-3.1-8b-instant' | 'mixtral-8x7b-32768';
  };
  coinmarketcap: {
    apiKey: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
    plan: 'free' | 'basic' | 'pro';
  };
  telegram: {
    botToken: string;
    chatId: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
  };
  rapidapi: {
    apiKey: string;
    enabled: boolean;
    status: 'connected' | 'disconnected' | 'untested';
    lastTested: string | null;
  };
}

// Default Configuration
const DEFAULT_API_KEYS: APIKeys = {
  binance: {
    apiKey: '',
    secretKey: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
    testnet: false,
  },
  okx: {
    apiKey: '',
    secretKey: '',
    passphrase: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
    testnet: false,
  },
  bybit: {
    apiKey: '',
    secretKey: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
    testnet: false,
  },
  btcturk: {
    apiKey: '',
    secretKey: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
  },
  groq: {
    apiKey: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
    model: 'llama-3.3-70b-versatile',
  },
  coinmarketcap: {
    apiKey: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
    plan: 'free',
  },
  telegram: {
    botToken: '',
    chatId: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
  },
  rapidapi: {
    apiKey: '',
    enabled: false,
    status: 'untested',
    lastTested: null,
  },
};

// Database storage (persistent, encrypted)
// Old: const apiKeysStore = new Map<string, APIKeys>();
// Now using: apiKeysDB from @/lib/database

/**
 * Get session ID from cookies
 */
async function getSessionId(_request: NextRequest): Promise<string> {
  const cookieStore = await cookies();
  let sessionId = cookieStore.get('session_id')?.value;

  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  return sessionId;
}

/**
 * Mask sensitive data (show only last 4 characters)
 */
function maskKey(key: string): string {
  if (!key || key.length <= 4) return key ? '***' : '';
  return '***' + key.slice(-4);
}

/**
 * Validate API key formats
 */
function validateAPIKeys(keys: Partial<APIKeys>): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Validate Binance
  if (keys.binance?.enabled) {
    if (!keys.binance.apiKey || keys.binance.apiKey.length < 20) {
      errors.push('Binance API Key must be at least 20 characters');
    }
    if (!keys.binance.secretKey || keys.binance.secretKey.length < 20) {
      errors.push('Binance Secret Key must be at least 20 characters');
    }
  }

  // Validate Groq
  if (keys.groq?.enabled) {
    if (!keys.groq.apiKey || !keys.groq.apiKey.startsWith('gsk_')) {
      errors.push('Groq API Key must start with "gsk_"');
    }
  }

  // Validate CoinMarketCap
  if (keys.coinmarketcap?.enabled) {
    if (!keys.coinmarketcap.apiKey || keys.coinmarketcap.apiKey.length < 30) {
      errors.push('CoinMarketCap API Key must be at least 30 characters');
    }
  }

  // Validate Telegram
  if (keys.telegram?.enabled) {
    if (!keys.telegram.botToken || !keys.telegram.botToken.includes(':')) {
      errors.push('Telegram Bot Token format invalid (should contain ":")');
    }
    if (!keys.telegram.chatId) {
      errors.push('Telegram Chat ID is required');
    }
  }

  // Validate RapidAPI
  if (keys.rapidapi?.enabled) {
    if (!keys.rapidapi.apiKey || keys.rapidapi.apiKey.length < 20) {
      errors.push('RapidAPI Key must be at least 20 characters');
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Test OKX API Connection
 */
async function testOKXConnection(
  apiKey: string,
  secretKey: string,
  passphrase: string,
  testnet: boolean
): Promise<{ success: boolean; message: string }> {
  try {
    const baseUrl = testnet ? 'https://www.okx.com' : 'https://www.okx.com';
    const timestamp = new Date().toISOString();
    const method = 'GET';
    const requestPath = '/api/v5/account/balance';

    const prehash = timestamp + method + requestPath;
    const signature = crypto.createHmac('sha256', secretKey).update(prehash).digest('base64');

    const response = await fetch(`${baseUrl}${requestPath}`, {
      headers: {
        'OK-ACCESS-KEY': apiKey,
        'OK-ACCESS-SIGN': signature,
        'OK-ACCESS-TIMESTAMP': timestamp,
        'OK-ACCESS-PASSPHRASE': passphrase,
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        message: `OKX connected successfully! Code: ${data.code || 0}`,
      };
    } else {
      const error = await response.json();
      return {
        success: false,
        message: `OKX error: ${error.msg || response.statusText}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test Bybit API Connection
 */
async function testBybitConnection(
  apiKey: string,
  secretKey: string,
  testnet: boolean
): Promise<{ success: boolean; message: string }> {
  try {
    const baseUrl = testnet ? 'https://api-testnet.bybit.com' : 'https://api.bybit.com';
    const timestamp = Date.now();
    const recvWindow = 5000;

    const queryString = `api_key=${apiKey}&timestamp=${timestamp}&recv_window=${recvWindow}`;
    const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

    const response = await fetch(`${baseUrl}/v2/private/wallet/balance?${queryString}&sign=${signature}`, {
      headers: {
        'Content-Type': 'application/json',
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        message: `Bybit connected successfully! Code: ${data.ret_code || 0}`,
      };
    } else {
      const error = await response.json();
      return {
        success: false,
        message: `Bybit error: ${error.ret_msg || response.statusText}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test BTCTurk API Connection
 */
async function testBTCTurkConnection(
  apiKey: string,
  secretKey: string
): Promise<{ success: boolean; message: string }> {
  try {
    const baseUrl = 'https://api.btcturk.com';
    const timestamp = Date.now();
    const _method = 'GET';
    const requestPath = '/api/v1/users/balances';

    const message = `${apiKey}${timestamp}`;
    const signature = crypto.createHmac('sha256', Buffer.from(secretKey, 'base64')).update(message).digest('base64');

    const response = await fetch(`${baseUrl}${requestPath}`, {
      headers: {
        'X-PCK': apiKey,
        'X-Stamp': timestamp.toString(),
        'X-Signature': signature,
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const _data = await response.json();
      return {
        success: true,
        message: `BTCTurk connected successfully!`,
      };
    } else {
      const error = await response.json();
      return {
        success: false,
        message: `BTCTurk error: ${error.message || response.statusText}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test Binance API Connection
 */
async function testBinanceConnection(
  apiKey: string,
  secretKey: string,
  testnet: boolean
): Promise<{ success: boolean; message: string }> {
  try {
    const baseUrl = testnet
      ? 'https://testnet.binancefuture.com'
      : 'https://fapi.binance.com';

    const timestamp = Date.now();
    const queryString = `timestamp=${timestamp}`;

    // Create signature
    const signature = crypto.createHmac('sha256', secretKey).update(queryString).digest('hex');

    const response = await fetch(`${baseUrl}/fapi/v2/account?${queryString}&signature=${signature}`, {
      headers: {
        'X-MBX-APIKEY': apiKey,
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        message: `Connected successfully! Balance: ${data.totalWalletBalance || 0} USDT`,
      };
    } else {
      const error = await response.json();
      return {
        success: false,
        message: `Binance error: ${error.msg || response.statusText}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test Groq AI Connection
 */
async function testGroqConnection(apiKey: string): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(_modelsUrl, {
      headers: {
        Authorization: `Bearer ${apiKey}`,
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        message: `Connected! Available models: ${data.data?.length || 0}`,
      };
    } else {
      return {
        success: false,
        message: `Groq error: ${response.status}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test CoinMarketCap Connection
 */
async function testCoinMarketCapConnection(apiKey: string): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch('https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest?limit=1', {
      headers: {
        'X-CMC_PRO_API_KEY': apiKey,
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      const data = await response.json();
      return {
        success: true,
        message: `Connected! Credits remaining: ${data.status?.credit_count || 'N/A'}`,
      };
    } else {
      const error = await response.json();
      return {
        success: false,
        message: `CoinMarketCap error: ${error.status?.error_message || response.statusText}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test Telegram Bot Connection
 */
async function testTelegramConnection(
  botToken: string,
  chatId: string
): Promise<{ success: boolean; message: string }> {
  try {
    // First, test if bot is valid
    const botResponse = await fetch(`https://api.telegram.org/bot${botToken}/getMe`, {
      signal: AbortSignal.timeout(5000),
    });

    if (!botResponse.ok) {
      return { success: false, message: 'Invalid bot token' };
    }

    const botData = await botResponse.json();

    // Test sending a message
    const messageResponse = await fetch(`https://api.telegram.org/bot${botToken}/sendMessage`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        chat_id: chatId,
        text: '✅ Telegram API test successful! Bot is connected.',
      }),
      signal: AbortSignal.timeout(5000),
    });

    if (messageResponse.ok) {
      return {
        success: true,
        message: `Connected to bot: @${botData.result?.username || 'unknown'}`,
      };
    } else {
      const error = await messageResponse.json();
      return {
        success: false,
        message: `Send message failed: ${error.description || 'Unknown error'}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * Test RapidAPI Connection
 */
async function testRapidAPIConnection(apiKey: string): Promise<{ success: boolean; message: string }> {
  try {
    // Test with a simple endpoint (example: Coinranking API)
    const response = await fetch('https://coinranking1.p.rapidapi.com/coins?limit=1', {
      headers: {
        'X-RapidAPI-Key': apiKey,
        'X-RapidAPI-Host': 'coinranking1.p.rapidapi.com',
      },
      signal: AbortSignal.timeout(5000),
    });

    if (response.ok) {
      return {
        success: true,
        message: 'RapidAPI connection successful!',
      };
    } else {
      return {
        success: false,
        message: `RapidAPI error: ${response.status}`,
      };
    }
  } catch (error) {
    return {
      success: false,
      message: error instanceof Error ? error.message : 'Connection failed',
    };
  }
}

/**
 * GET - Retrieve API keys (masked)
 */
export async function GET(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);

    // Get keys or use defaults
    let keys = apiKeysDB.get(sessionId);
    if (!keys) {
      keys = DEFAULT_API_KEYS;
      apiKeysDB.set(sessionId, keys);
    }

    // Mask sensitive data
    const maskedKeys: APIKeys = {
      binance: {
        ...keys.binance,
        apiKey: maskKey(keys.binance.apiKey),
        secretKey: maskKey(keys.binance.secretKey),
      },
      okx: {
        ...keys.okx,
        apiKey: maskKey(keys.okx.apiKey),
        secretKey: maskKey(keys.okx.secretKey),
        passphrase: maskKey(keys.okx.passphrase),
      },
      bybit: {
        ...keys.bybit,
        apiKey: maskKey(keys.bybit.apiKey),
        secretKey: maskKey(keys.bybit.secretKey),
      },
      btcturk: {
        ...keys.btcturk,
        apiKey: maskKey(keys.btcturk.apiKey),
        secretKey: maskKey(keys.btcturk.secretKey),
      },
      groq: {
        ...keys.groq,
        apiKey: maskKey(keys.groq.apiKey),
      },
      coinmarketcap: {
        ...keys.coinmarketcap,
        apiKey: maskKey(keys.coinmarketcap.apiKey),
      },
      telegram: {
        ...keys.telegram,
        botToken: maskKey(keys.telegram.botToken),
      },
      rapidapi: {
        ...keys.rapidapi,
        apiKey: maskKey(keys.rapidapi.apiKey),
      },
    };

    return NextResponse.json({
      success: true,
      data: maskedKeys,
    });
  } catch (error) {
    console.error('[API Keys API] GET Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get API keys',
      },
      { status: 500 }
    );
  }
}

/**
 * POST - Update or test API keys
 */
export async function POST(request: NextRequest) {
  try {
    const sessionId = await getSessionId(request);
    const body = await request.json();

    // Handle test requests
    if (body.action === 'test') {
      const { service, ...credentials } = body;

      let result: { success: boolean; message: string };

      switch (service) {
        case 'binance':
          result = await testBinanceConnection(
            credentials.apiKey,
            credentials.secretKey,
            credentials.testnet || false
          );
          break;

        case 'okx':
          result = await testOKXConnection(
            credentials.apiKey,
            credentials.secretKey,
            credentials.passphrase,
            credentials.testnet || false
          );
          break;

        case 'bybit':
          result = await testBybitConnection(
            credentials.apiKey,
            credentials.secretKey,
            credentials.testnet || false
          );
          break;

        case 'btcturk':
          result = await testBTCTurkConnection(credentials.apiKey, credentials.secretKey);
          break;

        case 'groq':
          result = await testGroqConnection(credentials.apiKey);
          break;

        case 'coinmarketcap':
          result = await testCoinMarketCapConnection(credentials.apiKey);
          break;

        case 'telegram':
          result = await testTelegramConnection(credentials.botToken, credentials.chatId);
          break;

        case 'rapidapi':
          result = await testRapidAPIConnection(credentials.apiKey);
          break;

        default:
          return NextResponse.json({ success: false, message: 'Invalid service' });
      }

      // Update status if test was successful
      if (result.success) {
        const currentKeys = apiKeysDB.get(sessionId) || DEFAULT_API_KEYS;
        const updatedKeys = { ...currentKeys };

        if (service === 'binance') {
          updatedKeys.binance = {
            ...updatedKeys.binance,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'okx') {
          updatedKeys.okx = {
            ...updatedKeys.okx,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'bybit') {
          updatedKeys.bybit = {
            ...updatedKeys.bybit,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'btcturk') {
          updatedKeys.btcturk = {
            ...updatedKeys.btcturk,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'groq') {
          updatedKeys.groq = {
            ...updatedKeys.groq,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'coinmarketcap') {
          updatedKeys.coinmarketcap = {
            ...updatedKeys.coinmarketcap,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'telegram') {
          updatedKeys.telegram = {
            ...updatedKeys.telegram,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        } else if (service === 'rapidapi') {
          updatedKeys.rapidapi = {
            ...updatedKeys.rapidapi,
            status: 'connected',
            lastTested: new Date().toISOString(),
          };
        }

        apiKeysDB.set(sessionId, updatedKeys);
      }

      return NextResponse.json(result);
    }

    // Validate input
    const validation = validateAPIKeys(body);
    if (!validation.valid) {
      return NextResponse.json(
        {
          success: false,
          error: 'Invalid API keys',
          details: validation.errors,
        },
        { status: 400 }
      );
    }

    // Get current keys or defaults
    const currentKeys = apiKeysDB.get(sessionId) || DEFAULT_API_KEYS;

    // Merge with new keys (only update provided fields)
    const updatedKeys: APIKeys = {
      binance: { ...currentKeys.binance, ...(body.binance || {}) },
      okx: { ...currentKeys.okx, ...(body.okx || {}) },
      bybit: { ...currentKeys.bybit, ...(body.bybit || {}) },
      btcturk: { ...currentKeys.btcturk, ...(body.btcturk || {}) },
      groq: { ...currentKeys.groq, ...(body.groq || {}) },
      coinmarketcap: { ...currentKeys.coinmarketcap, ...(body.coinmarketcap || {}) },
      telegram: { ...currentKeys.telegram, ...(body.telegram || {}) },
      rapidapi: { ...currentKeys.rapidapi, ...(body.rapidapi || {}) },
    };

    // Save to database
    apiKeysDB.set(sessionId, updatedKeys);

    // Create response with Set-Cookie header
    const response = NextResponse.json({
      success: true,
      message: 'API keys updated successfully',
    });

    // Set session cookie
    response.cookies.set('session_id', sessionId, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60, // 7 days
    });

    return response;
  } catch (error) {
    console.error('[API Keys API] POST Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to update API keys',
      },
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

    // Reset to defaults
    apiKeysDB.set(sessionId, DEFAULT_API_KEYS);

    return NextResponse.json({
      success: true,
      message: 'API keys reset to defaults',
      data: DEFAULT_API_KEYS,
    });
  } catch (error) {
    console.error('[API Keys API] PUT Error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to reset API keys',
      },
      { status: 500 }
    );
  }
}
