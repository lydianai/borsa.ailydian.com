/**
 * 🏆 HAREM ALTIN API - Type Definitions
 * Live Turkish Gold Price Data
 *
 * API: https://rapidapi.com/harem-altin
 * Host: harem-altin-live-gold-price-data.p.rapidapi.com
 */

export interface HaremAltinPrice {
  name: string;           // e.g., "Gram Altın", "Çeyrek Altın"
  buying: number;         // Alış fiyatı (TL)
  selling: number;        // Satış fiyatı (TL)
  change: number;         // Değişim yüzdesi
  changeAmount: number;   // Değişim miktarı (TL)
  updateTime: string;     // Son güncelleme zamanı
}

export interface HaremAltinResponse {
  success: boolean;
  data: {
    gram: HaremAltinPrice;
    ceyrek: HaremAltinPrice;
    yarim: HaremAltinPrice;
    tam: HaremAltinPrice;
    cumhuriyet: HaremAltinPrice;
    ata: HaremAltinPrice;
    gremse: HaremAltinPrice;
    currency?: {
      usd: number;
      eur: number;
      gbp: number;
    };
  };
  timestamp: number;
}

export interface FormattedGoldPrice {
  symbol: string;         // e.g., "XAUTR", "GOLD/TRY"
  name: string;           // e.g., "Gram Altın"
  price: number;          // Current price in TL
  change24h: number;      // 24h change percentage
  buyPrice: number;       // Alış
  sellPrice: number;      // Satış
  lastUpdate: Date;
  category: 'gold';
  currency: 'TRY';
}
