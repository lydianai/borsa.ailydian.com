/**
 * TRADITIONAL MARKETS AI ANALYSIS API
 * AI + Quantum Strategy Analysis for Forex, Metals, Indices
 *
 * Provider-agnostic: Any OpenAI-compatible API via environment variables:
 * - AI_API_KEY or GROQ_API_KEY: Your API key
 * - AI_API_URL: API endpoint (default: Groq)
 * - AI_MODEL: Model name (default: llama-3.3-70b-versatile)
 */

import { NextRequest, NextResponse } from 'next/server';

// AI Configuration
const AI_API_URL = process.env.AI_API_URL || 'https://api.groq.com/openai/v1/chat/completions';
const AI_MODEL = process.env.AI_MODEL || 'llama-3.3-70b-versatile';
const AI_API_KEY = process.env.AI_API_KEY || process.env.GROQ_API_KEY || '';

export const dynamic = 'force-dynamic';
export const runtime = 'nodejs';

interface AssetAnalysisRequest {
  symbol: string;
  price: number;
  change24h: number;
  assetType: 'metal' | 'forex' | 'index';
}

/**
 * GET /api/traditional-markets-analysis/[symbol]
 * Analyzes traditional market assets with AI + Quantum scoring
 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ symbol: string }> }
) {
  try {
    const { symbol } = await params;

    // Get query params
    const searchParams = request.nextUrl.searchParams;
    const price = parseFloat(searchParams.get('price') || '0');
    const change24h = parseFloat(searchParams.get('change24h') || '0');
    const assetType = (searchParams.get('type') || 'unknown') as 'metal' | 'forex' | 'index';

    if (!symbol || !price) {
      return NextResponse.json(
        { success: false, error: 'Missing required parameters: symbol, price' },
        { status: 400 }
      );
    }

    const assetTypeMap = {
      metal: 'Değerli Metal (Precious Metal)',
      forex: 'Döviz Kuru (Forex Pair vs TRY)',
      index: 'Endeks (Index)',
    };

    const prompt = `Sen profesyonel finansal analist bir AI'sın. **${symbol}** için kısa ve net analiz yap:

📊 ASSET BİLGİLERİ:
- Asset Type: ${assetTypeMap[assetType] || assetType}
- Current Price: ${assetType === 'index' ? `$${price.toFixed(3)}` : `₺${price.toFixed(2)}`}
- 24h Change: ${change24h.toFixed(2)}%

${assetType === 'metal' ? `
📈 METAL PİYASA ANALİZİ:
- Kıymetli metal piyasası trendleri
- Destek/Direnç seviyeleri tahmini
- Risk değerlendirmesi (düşük/orta/yüksek)
` : assetType === 'forex' ? `
💱 FOREX ANALİZİ:
- TRY bazlı döviz kuru trendi
- Merkez bankası politika etkisi
- Kısa vadeli hareket tahmini
` : `
📊 ENDEKS ANALİZİ:
- Global piyasa sentiment
- Teknik seviyeler
- Risk-reward değerlendirmesi
`}

ÇIKTI FORMATI:
- 3-4 cümle, Türkçe
- Net ve profesyonel
- Trader'a yönelik pratik bilgi

ÖNEMLİ: Yatırım tavsiyesi değil, analiz paylaşıyorsun.`;

    let aiAnalysis = 'AI analizi şu anda kullanılamıyor.';

    if (AI_API_KEY) {
      const aiResponse = await fetch(AI_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${AI_API_KEY}`,
        },
        body: JSON.stringify({
          model: AI_MODEL,
          messages: [{ role: 'user', content: prompt }],
          temperature: 0.7,
          max_tokens: 400,
        }),
      });

      if (aiResponse.ok) {
        const data = await aiResponse.json();
        aiAnalysis = data.choices?.[0]?.message?.content || aiAnalysis;
      }
    }

    // Quantum Score Calculation (60-100 range, weighted by change and volatility)
    const baseScore = 60;
    const changeScore = Math.min(Math.abs(change24h) * 2, 25); // Max 25 points from change
    const directionBonus = change24h > 0 ? 10 : 0; // +10 for positive momentum
    const volatilityBonus = Math.abs(change24h) > 2 ? 5 : 0; // +5 for high volatility

    const quantumScore = Math.min(
      Math.floor(baseScore + changeScore + directionBonus + volatilityBonus),
      100
    );

    // Recommendation Logic
    let recommendation = 'BEKLE';
    if (change24h > 2) recommendation = 'AL';
    else if (change24h < -2) recommendation = 'SAT';

    // Strategy Signals (Simulated - Traditional markets için özel)
    const strategies: any[] = [];

    // Moving Average Strategy
    strategies.push({
      name: 'Moving Average Trend',
      signal: change24h > 1 ? 'BUY' : change24h < -1 ? 'SELL' : 'WAIT',
      confidence: Math.min(50 + Math.abs(change24h) * 5, 95),
      strength: Math.min(Math.ceil(Math.abs(change24h) / 2), 10),
      reasoning: change24h > 1 ? 'Yükseliş trendi güçlü' : change24h < -1 ? 'Düşüş trendi aktif' : 'Yatay hareket',
    });

    // Momentum Strategy
    const momentumStrength = Math.abs(change24h);
    strategies.push({
      name: 'Momentum Indicator',
      signal: momentumStrength > 2 ? (change24h > 0 ? 'BUY' : 'SELL') : 'NEUTRAL',
      confidence: Math.min(45 + momentumStrength * 7, 90),
      strength: Math.min(Math.ceil(momentumStrength / 1.5), 10),
      reasoning: momentumStrength > 2 ? `${change24h > 0 ? 'Güçlü alım' : 'Güçlü satım'} momentumu` : 'Momentum zayıf',
    });

    // Volatility Strategy
    strategies.push({
      name: 'Volatility Breakout',
      signal: Math.abs(change24h) > 3 ? 'WAIT' : change24h > 0 ? 'BUY' : 'NEUTRAL',
      confidence: 65,
      strength: Math.min(Math.ceil(Math.abs(change24h) / 2.5), 10),
      reasoning: Math.abs(change24h) > 3 ? 'Yüksek volatilite - dikkatli ol' : 'Normal volatilite',
    });

    // Risk Management Strategy
    strategies.push({
      name: 'Risk Assessment',
      signal: Math.abs(change24h) < 1 ? 'BUY' : 'WAIT',
      confidence: Math.abs(change24h) < 1 ? 75 : 55,
      strength: Math.abs(change24h) < 1 ? 8 : 5,
      reasoning: Math.abs(change24h) < 1 ? 'Düşük risk profili' : 'Orta-yüksek risk',
    });

    // Count signals
    const buyCount = strategies.filter((s) => s.signal === 'BUY').length;
    const sellCount = strategies.filter((s) => s.signal === 'SELL').length;
    const waitCount = strategies.filter((s) => s.signal === 'WAIT').length;
    const neutralCount = strategies.filter((s) => s.signal === 'NEUTRAL').length;

    // Overall score (average confidence)
    const overallScore = Math.floor(
      strategies.reduce((sum, s) => sum + s.confidence, 0) / strategies.length
    );

    return NextResponse.json({
      success: true,
      data: {
        symbol,
        price,
        change24h,
        assetType,
        aiAnalysis,
        groqAnalysis: aiAnalysis,
        quantumScore,
        recommendation,
        overallScore,
        strategies,
        buyCount,
        sellCount,
        waitCount,
        neutralCount,
        timestamp: new Date().toISOString(),
      },
    });
  } catch (error: any) {
    console.error('[Traditional Markets Analysis Error]', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Analysis failed' },
      { status: 500 }
    );
  }
}
