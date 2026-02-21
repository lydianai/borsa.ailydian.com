/**
 * 🧠 ADVANCED AI ANALYZER - SARDAG EMRAH OTONOM SİSTEM
 *
 * Strategy Engine C (Premium AI) ile gelişmiş piyasa analizi.
 * Strategy Engine A'ya ek olarak daha derinlemesine analiz için kullanılır.
 *
 * Kullanım Alanları:
 * - Market sentiment analizi
 * - Strateji performans değerlendirmesi
 * - Yeni parametre önerileri
 * - Coin-bazlı adaptif ağırlıklandırma
 * - Doğal dil raporları
 */

import Anthropic from '@anthropic-ai/sdk';

// Singleton AI instance
let aiClient: Anthropic | null = null;

// Model configuration from environment
const AI_MODEL = process.env.ADVANCED_AI_MODEL || 'claude-3-5-sonnet-20241022';

/**
 * Advanced AI client'ını başlat
 */
export function getAdvancedAIClient(): Anthropic {
  if (aiClient) {
    return aiClient;
  }

  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error('ANTHROPIC_API_KEY environment variable is required');
  }

  aiClient = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  console.log('✅ Advanced AI Engine: Client initialized');

  return aiClient;
}

/**
 * Market datasını Advanced AI ile analiz et
 */
export async function analyzeMarketWithAdvancedAI(marketData: {
  symbol?: string;
  coins?: Array<{
    symbol: string;
    price: number;
    volume24h: number;
    change24h: number;
  }>;
  strategyResults?: Array<{
    name: string;
    signal: string;
    confidence: number;
  }>;
}) {
  const client = getAdvancedAIClient();

  const prompt = `SARDAG EMRAH Otonom AI Agent olarak şu market datasını analiz et:

${JSON.stringify(marketData, null, 2)}

Görevlerin:
1. **Mevcut 15 stratejinin performansını değerlendir**
   - Hangi stratejiler en güçlü sinyaller veriyor?
   - Hangileri tutarsız veya zayıf?

2. **Her coin için en uygun strateji kombinasyonunu öner**
   - Coin'in volatilitesine göre
   - Hacim profiline göre
   - Market fazına göre

3. **Yeni parametre önerileri sun**
   - MA periyotları optimize edilebilir mi?
   - RSI seviyeleri ayarlanmalı mı?
   - Volume threshold'ları değiştirilmeli mi?

4. **Risk skorları hesapla**
   - Her coin için 0-100 risk skoru
   - Faktörleri açıkla

5. **JSON formatında detaylı rapor oluştur**

BEYAZ ŞAPKA KURALLARI:
- Sadece bilgilendirme amaçlı analiz
- Yatırım tavsiyesi değil
- Kullanıcının kendi araştırmasını yapması gerektiğini vurgula
- Geçmiş performans garantisi değil

Lütfen yanıtını şu JSON formatında ver:

\`\`\`json
{
  "summary": "Genel market özeti (1-2 paragraf)",
  "topStrategies": [
    {
      "name": "Conservative Buy Signal",
      "performance": "excellent|good|moderate|poor",
      "reason": "Neden bu performansı gösteriyor"
    }
  ],
  "coinAnalysis": [
    {
      "symbol": "BTCUSDT",
      "recommendedStrategies": ["Conservative Buy Signal", "Volume Spike"],
      "weights": { "Conservative Buy Signal": 1.5, "Volume Spike": 1.2 },
      "riskScore": 35,
      "riskFactors": ["Yüksek volatilite", "Güçlü destek seviyesi"]
    }
  ],
  "parameterSuggestions": [
    {
      "strategy": "MA Crossover Pullback",
      "parameter": "fastMA",
      "currentValue": 10,
      "suggestedValue": 12,
      "reason": "Daha az yanlış sinyal için"
    }
  ],
  "disclaimer": "Bu analiz sadece bilgilendirme amaçlıdır. Yatırım tavsiyesi değildir."
}
\`\`\``;

  try {
    const message = await client.messages.create({
      model: AI_MODEL,
      max_tokens: 4096,
      temperature: 0.3, // Düşük temperature = daha tutarlı
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    // Yanıtı parse et
    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    // JSON'u çıkar
    const jsonMatch = responseText.match(/```json\n([\s\S]+?)\n```/);
    if (jsonMatch && jsonMatch[1]) {
      return JSON.parse(jsonMatch[1]);
    }

    // Eğer JSON bulunamazsa ham metni döndür
    return {
      summary: responseText,
      raw: true,
    };
  } catch (error: any) {
    console.error('❌ Advanced AI Error:', error.message);

    throw new Error(`Advanced AI analysis failed: ${error.message}`);
  }
}

/**
 * Strateji performansını Advanced AI ile değerlendir
 */
export async function evaluateStrategyPerformance(strategyData: {
  strategyName: string;
  recentSignals: Array<{
    timestamp: string;
    signal: string;
    confidence: number;
    outcome?: 'success' | 'failure' | 'pending';
  }>;
  successRate: number;
  avgConfidence: number;
}) {
  const client = getAdvancedAIClient();

  const prompt = `SARDAG EMRAH Trading Strategy Performance Evaluator olarak şu stratejiyi analiz et:

**Strateji**: ${strategyData.strategyName}
**Başarı Oranı**: ${strategyData.successRate}%
**Ortalama Güven**: ${strategyData.avgConfidence}%

**Son Sinyaller**:
${JSON.stringify(strategyData.recentSignals, null, 2)}

Görevlerin:
1. Stratejinin genel performansını değerlendir
2. Güçlü ve zayıf yönlerini belirle
3. İyileştirme önerileri sun
4. Bu stratejiyi hangi coin'lerde kullanmalıyız?
5. Ağırlık önerisi (0.5 - 2.0 arası)

JSON formatında yanıt ver:

\`\`\`json
{
  "overall": "excellent|good|moderate|poor",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "improvements": ["...", "..."],
  "bestFor": ["BTCUSDT", "ETHUSDT"],
  "recommendedWeight": 1.2,
  "reasoning": "..."
}
\`\`\``;

  try {
    const message = await client.messages.create({
      model: AI_MODEL,
      max_tokens: 2048,
      temperature: 0.3,
      messages: [{ role: 'user', content: prompt }],
    });

    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    const jsonMatch = responseText.match(/```json\n([\s\S]+?)\n```/);
    if (jsonMatch && jsonMatch[1]) {
      return JSON.parse(jsonMatch[1]);
    }

    return { raw: responseText };
  } catch (error: any) {
    console.error('❌ Advanced AI Strategy Evaluation Error:', error.message);
    throw error;
  }
}

/**
 * Coin-bazlı adaptif ağırlıklar öner
 */
export async function suggestCoinWeights(coinData: {
  symbol: string;
  historicalPerformance: {
    [strategy: string]: {
      successRate: number;
      tradeCount: number;
    };
  };
  currentMarket: {
    volatility: number;
    volume24h: number;
    trend: string;
  };
}) {
  const client = getAdvancedAIClient();

  const prompt = `SARDAG EMRAH Adaptive Weight Calculator olarak şu coin için strateji ağırlıkları öner:

**Coin**: ${coinData.symbol}

**Geçmiş Performans**:
${JSON.stringify(coinData.historicalPerformance, null, 2)}

**Mevcut Market**:
- Volatilite: ${coinData.currentMarket.volatility}%
- 24h Hacim: $${coinData.currentMarket.volume24h.toLocaleString()}
- Trend: ${coinData.currentMarket.trend}

Her strateji için 0.5 - 2.0 arası ağırlık belirle.
Başarı oranına ve mevcut market koşullarına göre karar ver.

JSON formatı:

\`\`\`json
{
  "weights": {
    "Conservative Buy Signal": 1.5,
    "Breakout-Retest": 1.2,
    "Volume Spike": 0.8
  },
  "reasoning": {
    "Conservative Buy Signal": "Yüksek başarı oranı bu coin'de",
    "Breakout-Retest": "Orta performans, tutarlı",
    "Volume Spike": "Düşük başarı, ağırlığı azalt"
  }
}
\`\`\``;

  try {
    const message = await client.messages.create({
      model: AI_MODEL,
      max_tokens: 1536,
      temperature: 0.2,
      messages: [{ role: 'user', content: prompt }],
    });

    const responseText = message.content[0].type === 'text' ? message.content[0].text : '';

    const jsonMatch = responseText.match(/```json\n([\s\S]+?)\n```/);
    if (jsonMatch && jsonMatch[1]) {
      return JSON.parse(jsonMatch[1]);
    }

    return { weights: {}, reasoning: {} };
  } catch (error: any) {
    console.error('❌ Advanced AI Weight Suggestion Error:', error.message);
    throw error;
  }
}

/**
 * Sağlık kontrolü
 */
export async function checkAdvancedAIHealth() {
  try {
    const client = getAdvancedAIClient();

    // Basit test mesajı
    const message = await client.messages.create({
      model: AI_MODEL,
      max_tokens: 50,
      messages: [{ role: 'user', content: 'Test: Respond with "OK"' }],
    });

    return {
      status: 'healthy',
      model: AI_MODEL,
      response: message.content[0].type === 'text' ? message.content[0].text : '',
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      error: error.message,
    };
  }
}

// Backward compatibility aliases
export const getClaudeClient = getAdvancedAIClient;
export const analyzeMarketWithClaude = analyzeMarketWithAdvancedAI;
export const checkClaudeHealth = checkAdvancedAIHealth;

export default {
  getAdvancedAIClient,
  analyzeMarketWithAdvancedAI,
  evaluateStrategyPerformance,
  suggestCoinWeights,
  checkAdvancedAIHealth,
  // Legacy exports for backward compatibility
  getClaudeClient,
  analyzeMarketWithClaude,
  checkClaudeHealth,
};
