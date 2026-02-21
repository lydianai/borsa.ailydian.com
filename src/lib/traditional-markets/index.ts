/**
  * TRADITIONAL MARKETS - UNIFIED SERVICE
 * Aggregates all traditional market data - LIVE DATA ONLY
 *
 * Combines:
 * - Precious Metals (Gold, Silver, Palladium, Copper)
 * - Turkish Gold (Harem Altın API - 17 products)
 * - Forex (10 major currencies vs TRY)
 * - DXY (US Dollar Index)
 * - Energy (Brent, WTI, Natural Gas)
 * - Stock Indices (S&P 500, NASDAQ, Dow Jones)
 * - Treasury Bonds (US 2Y, 10Y, 30Y)
 * - Agriculture (Wheat, Corn, Soybeans, Coffee, Sugar)
 */

import {
  getPreciousMetalsData,
  clearPreciousMetalsCache,
  type PreciousMetalsData,
  type GoldPrice,
  type PreciousMetalPrice,
} from './precious-metals-adapter';

import {
  getForexData,
  getCurrencyRate,
  clearForexCache,
  type ForexData,
  type ForexRate,
} from './forex-adapter';

import {
  getDXYData,
  clearDXYCache,
  getDXYTrend,
  type DXYData,
} from './dxy-adapter';

import {
  fetchEnergyCommodities,
  clearEnergyCache,
  type EnergyCommodity,
} from '../adapters/energy-commodities-adapter';

import {
  fetchStockIndices,
  clearIndicesCache,
  type StockIndex,
} from '../adapters/stock-indices-adapter';

import {
  fetchTreasuryBonds,
  clearBondsCache,
  type TreasuryBond,
} from '../adapters/treasury-bonds-adapter';

import {
  fetchAgriculturalCommodities,
  clearAgricultureCache,
  type AgriculturalCommodity,
} from '../adapters/agricultural-commodities-adapter';

// ============================================================================
// RE-EXPORTS
// ============================================================================

export type {
  PreciousMetalsData,
  GoldPrice,
  PreciousMetalPrice,
  ForexData,
  ForexRate,
  DXYData,
  EnergyCommodity,
  StockIndex,
  TreasuryBond,
  AgriculturalCommodity,
};

export {
  // Precious Metals
  getPreciousMetalsData,
  clearPreciousMetalsCache,

  // Forex
  getForexData,
  getCurrencyRate,
  clearForexCache,

  // DXY
  getDXYData,
  clearDXYCache,
  getDXYTrend,

  // Energy
  fetchEnergyCommodities,
  clearEnergyCache,

  // Stock Indices
  fetchStockIndices,
  clearIndicesCache,

  // Treasury Bonds
  fetchTreasuryBonds,
  clearBondsCache,

  // Agriculture
  fetchAgriculturalCommodities,
  clearAgricultureCache,
};

// ============================================================================
// UNIFIED DATA TYPE
// ============================================================================

export interface TraditionalMarketsData {
  metals: PreciousMetalsData;
  forex: ForexData;
  dxy: DXYData;
  energy: EnergyCommodity[];
  stockIndices: StockIndex[];
  bonds: TreasuryBond[];
  agriculture: AgriculturalCommodity[];
  timestamp: Date;
  summary: {
    totalAssets: number;
    categories: {
      metals: number;
      currencies: number;
      indices: number;
      energy: number;
      bonds: number;
      agriculture: number;
    };
  };
}

// ============================================================================
// UNIFIED SERVICE
// ============================================================================

/**
 * Get all traditional markets data in one call
 * REAL-TIME DATA ONLY - NO MOCK DATA
 */
export async function getAllTraditionalMarketsData(
  forceRefresh: boolean = false
): Promise<TraditionalMarketsData> {
  console.log('[TraditionalMarkets] Fetching all data...');

  try {
    // Fetch all data in parallel for maximum performance
    const [metals, forex, dxy, energy, stockIndices, bonds, agriculture] = await Promise.all([
      getPreciousMetalsData(forceRefresh),
      getForexData(forceRefresh),
      getDXYData(forceRefresh),
      fetchEnergyCommodities(),
      fetchStockIndices(),
      fetchTreasuryBonds(),
      fetchAgriculturalCommodities(),
    ]);

    const totalIndices = 1 + stockIndices.length; // DXY + Stock Indices

    const data: TraditionalMarketsData = {
      metals,
      forex,
      dxy,
      energy,
      stockIndices,
      bonds,
      agriculture,
      timestamp: new Date(),
      summary: {
        totalAssets:
          4 + // 4 precious metals (Gold, Silver, Palladium, Copper)
          forex.rates.length + // Currency pairs
          1 + // DXY
          energy.length + // Energy commodities
          stockIndices.length + // Stock indices
          bonds.length + // Treasury bonds
          agriculture.length, // Agricultural commodities
        categories: {
          metals: 4, // Gold, Silver, Palladium, Copper
          currencies: forex.rates.length,
          indices: totalIndices, // DXY + Stock Indices
          energy: energy.length,
          bonds: bonds.length,
          agriculture: agriculture.length,
        },
      },
    };

    console.log(`[TraditionalMarkets] All data fetched successfully - ${data.summary.totalAssets} total assets`);
    return data;
  } catch (error: any) {
    console.error('[TraditionalMarkets] Failed to fetch all data:', error);
    throw new Error(`Failed to fetch traditional markets data: ${error.message}`);
  }
}

/**
 * Clear all traditional markets caches
 */
export function clearAllTraditionalMarketsCache(): void {
  clearPreciousMetalsCache();
  clearForexCache();
  clearDXYCache();
  clearEnergyCache();
  clearIndicesCache();
  clearBondsCache();
  clearAgricultureCache();
  console.log('[TraditionalMarkets] All caches cleared (7 categories)');
}

/**
 * Get specific asset data by symbol
 */
export async function getAssetBySymbol(
  symbol: string,
  forceRefresh: boolean = false
): Promise<
  | GoldPrice
  | PreciousMetalPrice
  | ForexRate
  | DXYData
  | EnergyCommodity
  | StockIndex
  | TreasuryBond
  | AgriculturalCommodity
  | null
> {
  // Check if it's a metal
  if (['XAU', 'XAG', 'XPD', 'XCU'].includes(symbol)) {
    const metals = await getPreciousMetalsData(forceRefresh);
    switch (symbol) {
      case 'XAU':
        return metals.gold;
      case 'XAG':
        return metals.silver;
      case 'XPD':
        return metals.palladium;
      case 'XCU':
        return metals.copper;
    }
  }

  // Check if it's a currency
  if (['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY', 'RUB', 'SAR'].includes(symbol)) {
    return await getCurrencyRate(symbol, forceRefresh);
  }

  // Check if it's DXY
  if (symbol === 'DXY') {
    return await getDXYData(forceRefresh);
  }

  // Check if it's energy
  if (['BRENT', 'WTI', 'NATGAS'].includes(symbol)) {
    const energy = await fetchEnergyCommodities();
    return energy.find((e) => e.symbol === symbol) || null;
  }

  // Check if it's a stock index
  if (['SPX', 'NDX', 'DJI'].includes(symbol)) {
    const indices = await fetchStockIndices();
    return indices.find((i) => i.symbol === symbol) || null;
  }

  // Check if it's a bond
  if (['US2Y', 'US10Y', 'US30Y'].includes(symbol)) {
    const bonds = await fetchTreasuryBonds();
    return bonds.find((b) => b.symbol === symbol) || null;
  }

  // Check if it's agriculture
  if (['WHEAT', 'CORN', 'SOYBEAN', 'COFFEE', 'SUGAR'].includes(symbol)) {
    const agriculture = await fetchAgriculturalCommodities();
    return agriculture.find((a) => a.symbol === symbol) || null;
  }

  return null;
}

/**
 * Get market overview summary
 */
export async function getMarketOverview(
  forceRefresh: boolean = false
): Promise<{
  trending: Array<{ symbol: string; name: string; change: number }>;
  strongest: { symbol: string; name: string; change: number } | null;
  weakest: { symbol: string; name: string; change: number } | null;
  dxyTrend: 'bullish' | 'bearish' | 'neutral';
}> {
  const data = await getAllTraditionalMarketsData(forceRefresh);

  // Collect all assets with changes
  const allAssets: Array<{ symbol: string; name: string; change: number }> = [
    // Precious Metals
    {
      symbol: data.metals.gold.symbol,
      name: data.metals.gold.name,
      change: data.metals.gold.change24h,
    },
    {
      symbol: data.metals.silver.symbol,
      name: data.metals.silver.name,
      change: data.metals.silver.change24h,
    },
    {
      symbol: data.metals.palladium.symbol,
      name: data.metals.palladium.name,
      change: data.metals.palladium.change24h,
    },
    {
      symbol: data.metals.copper.symbol,
      name: data.metals.copper.name,
      change: data.metals.copper.change24h,
    },
    // Forex
    ...data.forex.rates.map((rate) => ({
      symbol: rate.symbol,
      name: rate.name,
      change: rate.change24h,
    })),
    // DXY
    {
      symbol: data.dxy.symbol,
      name: data.dxy.name,
      change: data.dxy.changePercent,
    },
    // Energy Commodities
    ...data.energy.map((commodity) => ({
      symbol: commodity.symbol,
      name: commodity.name,
      change: commodity.change24h,
    })),
    // Stock Indices
    ...data.stockIndices.map((index) => ({
      symbol: index.symbol,
      name: index.name,
      change: index.changePercent,
    })),
    // Treasury Bonds
    ...data.bonds.map((bond) => ({
      symbol: bond.symbol,
      name: bond.name,
      change: bond.change24h,
    })),
    // Agricultural Commodities
    ...data.agriculture.map((agri) => ({
      symbol: agri.symbol,
      name: agri.name,
      change: agri.change24h,
    })),
  ];

  // Sort by absolute change
  const sorted = [...allAssets].sort((a, b) => Math.abs(b.change) - Math.abs(a.change));

  // Get top movers
  const trending = sorted.slice(0, 10); // Increased from 5 to 10 due to more assets

  // Get strongest/weakest
  const byChange = [...allAssets].sort((a, b) => b.change - a.change);
  const strongest = byChange[0] || null;
  const weakest = byChange[byChange.length - 1] || null;

  return {
    trending,
    strongest,
    weakest,
    dxyTrend: getDXYTrend(data.dxy),
  };
}
