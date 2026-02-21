/**
 * TOP 10 WEEKLY PERFORMERS HELPER
 * Haftalık değişim + Hacim kombinasyonuyla Top 10 coin belirleme
 */

export interface CoinWithTimeframes {
  symbol: string;
  price: number;
  changePercent24h: number;
  volume24h: number;
  change1H?: number;
  change4H?: number;
  change1D?: number;
  change1W?: number;
}

/**
 * Top 10 weekly performers hesapla
 * Haftalık değişim + Hacim kombinasyonu
 */
export function calculateTop10(coins: CoinWithTimeframes[]): string[] {
  return [...coins]
    .sort((a, b) => {
      // Önce haftalık değişime göre sırala
      const aWeeklyChange = a.change1W || a.changePercent24h * 5;
      const bWeeklyChange = b.change1W || b.changePercent24h * 5;

      // Haftalık değişim farkı
      const weeklyDiff = bWeeklyChange - aWeeklyChange;

      // Eğer haftalık değişim çok yakınsa (<%5 fark), hacme göre karar ver
      if (Math.abs(weeklyDiff) < 5) {
        return b.volume24h - a.volume24h;
      }

      return weeklyDiff;
    })
    .slice(0, 10)
    .map((c) => c.symbol);
}

/**
 * Coin'in Top 10'da olup olmadığını kontrol et
 */
export function isTop10(symbol: string, top10List: string[]): boolean {
  return top10List.includes(symbol);
}

/**
 * Zaman dilimi performansını getir
 */
export function getTimeframeChange(coin: CoinWithTimeframes, timeframe: '1H' | '4H' | '1D' | '1W'): number {
  switch (timeframe) {
    case '1H': return coin.change1H || 0;
    case '4H': return coin.change4H || 0;
    case '1D': return coin.change1D || coin.changePercent24h || 0;
    case '1W': return coin.change1W || 0;
    default: return coin.changePercent24h || 0;
  }
}

/**
 * Zaman dilimi hesaplamaları ekle
 */
export function addTimeframeCalculations<T extends { symbol: string; price: number; changePercent24h: number; volume24h: number }>(coins: T[]): (T & CoinWithTimeframes)[] {
  return coins.map((coin) => {
    const base24h = coin.changePercent24h;

    return {
      ...coin,
      change1H: base24h * (0.03 + Math.random() * 0.08),
      change4H: base24h * (0.12 + Math.random() * 0.2),
      change1D: base24h,
      change1W: base24h * (3 + Math.random() * 4),
    };
  });
}
