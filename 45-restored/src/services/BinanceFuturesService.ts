import axios from 'axios';
import crypto from 'crypto';

interface FuturesSymbol {
  symbol: string;
  pair: string;
  baseAsset: string;
  quoteAsset: string;
  status: string;
}

interface FuturesTicker {
  symbol: string;
  priceChange: string;
  priceChangePercent: string;
  lastPrice: string;
  volume: string;
  quoteVolume: string;
  highPrice: string;
  lowPrice: string;
}

interface FuturesKline {
  openTime: number;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: number;
}

export class BinanceFuturesService {
  private baseUrl = 'https://fapi.binance.com';
  private apiKey: string | null = null;
  private apiSecret: string | null = null;

  constructor(apiKey?: string, apiSecret?: string) {
    this.apiKey = apiKey || null;
    this.apiSecret = apiSecret || null;
  }

  setCredentials(apiKey: string, apiSecret: string) {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
  }

  private createSignature(queryString: string): string {
    if (!this.apiSecret) throw new Error('API Secret not set');
    return crypto
      .createHmac('sha256', this.apiSecret)
      .update(queryString)
      .digest('hex');
  }

  private async request(endpoint: string, params: any = {}, signed: boolean = false) {
    try {
      let queryString = Object.keys(params)
        .map(key => `${key}=${encodeURIComponent(params[key])}`)
        .join('&');

      if (signed) {
        if (!this.apiKey || !this.apiSecret) {
          throw new Error('API credentials required for signed requests');
        }
        const timestamp = Date.now();
        queryString += `&timestamp=${timestamp}`;
        const signature = this.createSignature(queryString);
        queryString += `&signature=${signature}`;
      }

      const url = `${this.baseUrl}${endpoint}${queryString ? '?' + queryString : ''}`;
      const headers: any = {};
      if (this.apiKey) {
        headers['X-MBX-APIKEY'] = this.apiKey;
      }

      const response = await axios.get(url, { headers, timeout: 10000 });
      return response.data;
    } catch (error: any) {
      console.error(`Binance Futures API Error [${endpoint}]:`, error.message);
      throw error;
    }
  }

  async getAllUSDTFuturesSymbols(): Promise<FuturesSymbol[]> {
    try {
      const exchangeInfo = await this.request('/fapi/v1/exchangeInfo');
      return exchangeInfo.symbols
        .filter((s: any) => s.quoteAsset === 'USDT' && s.status === 'TRADING')
        .map((s: any) => ({
          symbol: s.symbol,
          pair: s.pair,
          baseAsset: s.baseAsset,
          quoteAsset: s.quoteAsset,
          status: s.status
        }));
    } catch (error) {
      console.error('Error fetching USDT futures symbols:', error);
      return [];
    }
  }

  async get24hrTickers(): Promise<FuturesTicker[]> {
    try {
      const tickers = await this.request('/fapi/v1/ticker/24hr');
      return tickers
        .filter((t: any) => t.symbol.endsWith('USDT'))
        .map((t: any) => ({
          symbol: t.symbol,
          priceChange: t.priceChange,
          priceChangePercent: t.priceChangePercent,
          lastPrice: t.lastPrice,
          volume: t.volume,
          quoteVolume: t.quoteVolume,
          highPrice: t.highPrice,
          lowPrice: t.lowPrice
        }));
    } catch (error) {
      console.error('Error fetching 24hr tickers:', error);
      return [];
    }
  }

  async getTopMovers(limit: number = 20): Promise<FuturesTicker[]> {
    try {
      const tickers = await this.get24hrTickers();
      return tickers
        .sort((a, b) => Math.abs(parseFloat(b.priceChangePercent)) - Math.abs(parseFloat(a.priceChangePercent)))
        .slice(0, limit);
    } catch (error) {
      console.error('Error fetching top movers:', error);
      return [];
    }
  }

  async getKlines(symbol: string, interval: string = '15m', limit: number = 100): Promise<FuturesKline[]> {
    try {
      const klines = await this.request('/fapi/v1/klines', { symbol, interval, limit });
      return klines.map((k: any) => ({
        openTime: k[0],
        open: k[1],
        high: k[2],
        low: k[3],
        close: k[4],
        volume: k[5],
        closeTime: k[6]
      }));
    } catch (error) {
      console.error(`Error fetching klines for ${symbol}:`, error);
      return [];
    }
  }

  async getAccountInfo() {
    try {
      return await this.request('/fapi/v2/account', {}, true);
    } catch (error) {
      console.error('Error fetching account info:', error);
      throw error;
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.request('/fapi/v1/ping');
      return true;
    } catch (error) {
      return false;
    }
  }

  async testAPIKeys(): Promise<{ valid: boolean; message: string }> {
    try {
      if (!this.apiKey || !this.apiSecret) {
        return { valid: false, message: 'API anahtarları ayarlanmamış' };
      }

      await this.getAccountInfo();
      return { valid: true, message: 'API anahtarları geçerli' };
    } catch (error: any) {
      if (error.response?.status === 401) {
        return { valid: false, message: 'API anahtarları geçersiz' };
      }
      return { valid: false, message: error.message || 'Bağlantı hatası' };
    }
  }
}

export const binanceFuturesService = new BinanceFuturesService();
