import crypto from 'crypto';

export interface FuturesPosition {
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  leverage: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  liquidationPrice: number;
  margin: number;
}

export interface FuturesOrder {
  symbol: string;
  side: 'BUY' | 'SELL';
  type: 'LIMIT' | 'MARKET' | 'STOP_MARKET' | 'TAKE_PROFIT_MARKET';
  quantity: number;
  price?: number;
  stopPrice?: number;
  timeInForce?: 'GTC' | 'IOC' | 'FOK';
  reduceOnly?: boolean;
}

export interface FuturesBalance {
  asset: string;
  balance: number;
  availableBalance: number;
  unrealizedProfit: number;
}

export class BinanceFuturesAPI {
  private apiKey: string;
  private apiSecret: string;
  private baseUrl: string = 'https://fapi.binance.com';
  private _testnet: boolean;

  constructor(apiKey: string, apiSecret: string, testnet: boolean = false) {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    this.testnet = testnet;

    if (testnet) {
      this.baseUrl = 'https://testnet.binancefuture.com';
    }
  }

  private createSignature(queryString: string): string {
    return crypto
      .createHmac('sha256', this.apiSecret)
      .update(queryString)
      .digest('hex');
  }

  private async request(
    endpoint: string,
    method: 'GET' | 'POST' | 'DELETE' = 'GET',
    params: Record<string, any> = {},
    signed: boolean = false
  ): Promise<any> {
    const timestamp = Date.now();

    if (signed) {
      params.timestamp = timestamp;
      params.recvWindow = 5000;
    }

    const queryString = Object.keys(params)
      .map(key => `${key}=${params[key]}`)
      .join('&');

    const signature = signed ? this.createSignature(queryString) : '';
    const url = signed
      ? `${this.baseUrl}${endpoint}?${queryString}&signature=${signature}`
      : `${this.baseUrl}${endpoint}?${queryString}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      headers['X-MBX-APIKEY'] = this.apiKey;
    }

    const response = await fetch(url, {
      method,
      headers,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`Binance API Error: ${error.msg || error.code}`);
    }

    return response.json();
  }

  // Test API connectivity
  async ping(): Promise<boolean> {
    try {
      await this.request('/fapi/v1/ping');
      return true;
    } catch {
      return false;
    }
  }

  // Get server time
  async getServerTime(): Promise<number> {
    const response = await this.request('/fapi/v1/time');
    return response.serverTime;
  }

  // Get exchange info
  async getExchangeInfo(): Promise<any> {
    return this.request('/fapi/v1/exchangeInfo');
  }

  // Get account balance
  async getBalance(): Promise<FuturesBalance[]> {
    const response = await this.request('/fapi/v2/balance', 'GET', {}, true);
    return response.map((item: any) => ({
      asset: item.asset,
      balance: parseFloat(item.balance),
      availableBalance: parseFloat(item.availableBalance),
      unrealizedProfit: parseFloat(item.crossUnPnl),
    }));
  }

  // Get account information
  async getAccountInfo(): Promise<any> {
    return this.request('/fapi/v2/account', 'GET', {}, true);
  }

  // Get current positions
  async getPositions(): Promise<FuturesPosition[]> {
    const response = await this.request('/fapi/v2/positionRisk', 'GET', {}, true);

    return response
      .filter((pos: any) => parseFloat(pos.positionAmt) !== 0)
      .map((pos: any) => ({
        symbol: pos.symbol,
        side: parseFloat(pos.positionAmt) > 0 ? 'LONG' : 'SHORT',
        entryPrice: parseFloat(pos.entryPrice),
        currentPrice: parseFloat(pos.markPrice),
        quantity: Math.abs(parseFloat(pos.positionAmt)),
        leverage: parseInt(pos.leverage),
        unrealizedPnl: parseFloat(pos.unRealizedProfit),
        unrealizedPnlPercent: (parseFloat(pos.unRealizedProfit) / parseFloat(pos.notional)) * 100,
        liquidationPrice: parseFloat(pos.liquidationPrice),
        margin: parseFloat(pos.isolatedMargin),
      }));
  }

  // Change leverage
  async changeLeverage(symbol: string, leverage: number): Promise<any> {
    return this.request('/fapi/v1/leverage', 'POST', { symbol, leverage }, true);
  }

  // Change margin type (ISOLATED or CROSS)
  async changeMarginType(symbol: string, marginType: 'ISOLATED' | 'CROSS'): Promise<any> {
    return this.request('/fapi/v1/marginType', 'POST', { symbol, marginType }, true);
  }

  // Place new order
  async placeOrder(order: FuturesOrder): Promise<any> {
    const params: Record<string, any> = {
      symbol: order.symbol,
      side: order.side,
      type: order.type,
      quantity: order.quantity,
    };

    if (order.price) {
      params.price = order.price;
    }

    if (order.stopPrice) {
      params.stopPrice = order.stopPrice;
    }

    if (order.timeInForce) {
      params.timeInForce = order.timeInForce;
    }

    if (order.reduceOnly !== undefined) {
      params.reduceOnly = order.reduceOnly;
    }

    return this.request('/fapi/v1/order', 'POST', params, true);
  }

  // Cancel order
  async cancelOrder(symbol: string, orderId: number): Promise<any> {
    return this.request('/fapi/v1/order', 'DELETE', { symbol, orderId }, true);
  }

  // Cancel all open orders
  async cancelAllOrders(symbol: string): Promise<any> {
    return this.request('/fapi/v1/allOpenOrders', 'DELETE', { symbol }, true);
  }

  // Get open orders
  async getOpenOrders(symbol?: string): Promise<any[]> {
    const params = symbol ? { symbol } : {};
    return this.request('/fapi/v1/openOrders', 'GET', params, true);
  }

  // Get order history
  async getOrderHistory(symbol: string, limit: number = 500): Promise<any[]> {
    return this.request('/fapi/v1/allOrders', 'GET', { symbol, limit }, true);
  }

  // Get current price
  async getPrice(symbol: string): Promise<number> {
    const response = await this.request('/fapi/v1/ticker/price', 'GET', { symbol });
    return parseFloat(response.price);
  }

  // Get 24hr ticker
  async get24hrTicker(symbol: string): Promise<any> {
    return this.request('/fapi/v1/ticker/24hr', 'GET', { symbol });
  }

  // Close position (market order to reduce position to 0)
  async closePosition(symbol: string, side: 'LONG' | 'SHORT'): Promise<any> {
    const positions = await this.getPositions();
    const position = positions.find(p => p.symbol === symbol && p.side === side);

    if (!position) {
      throw new Error(`No ${side} position found for ${symbol}`);
    }

    // Close LONG = SELL, Close SHORT = BUY
    const closeSide = side === 'LONG' ? 'SELL' : 'BUY';

    return this.placeOrder({
      symbol,
      side: closeSide,
      type: 'MARKET',
      quantity: position.quantity,
      reduceOnly: true,
    });
  }

  // Set stop-loss
  async setStopLoss(
    symbol: string,
    side: 'LONG' | 'SHORT',
    stopPrice: number,
    quantity: number
  ): Promise<any> {
    // For LONG position, stop-loss is a SELL order below current price
    // For SHORT position, stop-loss is a BUY order above current price
    const orderSide = side === 'LONG' ? 'SELL' : 'BUY';

    return this.placeOrder({
      symbol,
      side: orderSide,
      type: 'STOP_MARKET',
      quantity,
      stopPrice,
      reduceOnly: true,
    });
  }

  // Set take-profit
  async setTakeProfit(
    symbol: string,
    side: 'LONG' | 'SHORT',
    takeProfitPrice: number,
    quantity: number
  ): Promise<any> {
    // For LONG position, take-profit is a SELL order above current price
    // For SHORT position, take-profit is a BUY order below current price
    const orderSide = side === 'LONG' ? 'SELL' : 'BUY';

    return this.placeOrder({
      symbol,
      side: orderSide,
      type: 'TAKE_PROFIT_MARKET',
      quantity,
      stopPrice: takeProfitPrice,
      reduceOnly: true,
    });
  }
}
