import { NextResponse } from 'next/server';

// CoinGecko API - Free tier, no API key required
const COINGECKO_API = 'https://api.coingecko.com/api/v3';

export async function GET() {
  try {
    // Fetch top cryptocurrencies
    const response = await fetch(
      `${COINGECKO_API}/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=50&page=1&sparkline=true&price_change_percentage=1h,24h,7d`,
      {
        headers: {
          'Accept': 'application/json',
        },
        next: { revalidate: 60 } // Cache for 1 minute
      }
    );

    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.status}`);
    }

    const data = await response.json();

    // Transform data
    const cryptos = data.map((coin: any) => ({
      id: coin.id,
      symbol: coin.symbol.toUpperCase(),
      name: coin.name,
      image: coin.image,
      currentPrice: coin.current_price,
      marketCap: coin.market_cap,
      marketCapRank: coin.market_cap_rank,
      totalVolume: coin.total_volume,
      priceChange1h: coin.price_change_percentage_1h_in_currency || 0,
      priceChange24h: coin.price_change_percentage_24h || 0,
      priceChange7d: coin.price_change_percentage_7d_in_currency || 0,
      high24h: coin.high_24h,
      low24h: coin.low_24h,
      circulatingSupply: coin.circulating_supply,
      totalSupply: coin.total_supply,
      maxSupply: coin.max_supply,
      ath: coin.ath,
      athDate: coin.ath_date,
      sparkline: coin.sparkline_in_7d?.price || [],
      lastUpdated: coin.last_updated
    }));

    console.log(`✅ Fetched ${cryptos.length} cryptocurrencies from CoinGecko`);

    return NextResponse.json({
      success: true,
      data: cryptos,
      count: cryptos.length,
      timestamp: new Date().toISOString(),
      source: 'CoinGecko API v3'
    }, {
      headers: {
        'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=120',
      }
    });

  } catch (error: any) {
    console.error('❌ Crypto API error:', error);

    return NextResponse.json({
      success: false,
      error: error.message,
      data: [],
      timestamp: new Date().toISOString()
    }, {
      status: 500
    });
  }
}

// Get single crypto details
export async function POST(request: Request) {
  try {
    const { coinId } = await request.json();

    if (!coinId) {
      return NextResponse.json(
        { error: 'Coin ID required' },
        { status: 400 }
      );
    }

    const response = await fetch(
      `${COINGECKO_API}/coins/${coinId}?localization=false&tickers=false&community_data=false&developer_data=false`,
      {
        headers: {
          'Accept': 'application/json',
        },
        next: { revalidate: 60 }
      }
    );

    if (!response.ok) {
      throw new Error(`CoinGecko API error: ${response.status}`);
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      data: {
        id: data.id,
        symbol: data.symbol.toUpperCase(),
        name: data.name,
        description: data.description?.en || '',
        image: data.image?.large,
        marketData: data.market_data,
        links: data.links
      }
    });

  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
}