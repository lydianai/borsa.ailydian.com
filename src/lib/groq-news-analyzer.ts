/**
 * AI News Analyzer & Translator
 *
 * Features:
 * 1. English -> Turkish translation
 * 2. Impact scoring (1-10)
 * 3. Category classification
 * 4. Sentiment analysis
 * 5. Tag generation
 *
 * Filters only IMPORTANT news (impact >= 7)
 *
 * Supports any OpenAI-compatible API via environment variables:
 * - AI_API_KEY or GROQ_API_KEY: Your API key
 * - AI_API_URL: API endpoint (default: Groq)
 * - NEWS_AI_MODEL: Model name (default: llama-3.3-70b-versatile)
 */

import type { GroqNewsAnalysisResponse } from '@/types/rapid-api';

// Configuration
const AI_API_URL = process.env.AI_API_URL || 'https://api.groq.com/openai/v1/chat/completions';
const AI_MODEL = process.env.NEWS_AI_MODEL || 'llama-3.3-70b-versatile';
const AI_API_KEY = process.env.AI_API_KEY || process.env.GROQ_API_KEY || '';

interface NewsInput {
  title: string;
  description: string;
  source: string;
}

const SYSTEM_PROMPT = `Sen bir kripto para uzmanısın. Görevin:
1. Kripto haber başlık ve açıklamalarını Türkçe'ye çevirmek
2. Haberin piyasa etkisini 1-10 arası puanlamak (10=çok kritik, 1=önemsiz)
3. Haberi kategorize etmek
4. Sentiment analizi yapmak
5. İlgili tag'ler oluşturmak

PUANLAMA KRİTERLERİ:
- 9-10: Bitcoin/Ethereum major news, SEC düzenlemeleri, büyük hack'ler, major exchange sorunları
- 7-8: Önemli protokol güncellemeleri, büyük ortaklıklar, önemli ekonomik veriler
- 5-6: Orta seviye haberler, minor güncellemeler
- 1-4: Küçük haberler, spam, minor token haberleri

SADECE 7 VE ÜZERİ PUANLI HABERLERİ FİLTRELE!

KATEGORİLER:
- bitcoin: Bitcoin ile ilgili
- ethereum: Ethereum ile ilgili
- regulation: Yasal/düzenleme haberleri
- defi: DeFi protokolleri
- market: Piyasa analizi/ekonomi
- technology: Teknoloji/güncelleme
- other: Diğer

CEVAP FORMATI (JSON):
{
  "titleTR": "Türkçe başlık",
  "descriptionTR": "Türkçe açıklama",
  "impactScore": 8,
  "category": "bitcoin",
  "sentiment": "positive",
  "tags": ["Bitcoin", "Fiyat", "Artış"],
  "reasoning": "Neden bu puan verildi?"
}`;

/**
 * Send chat completion request
 */
async function chatCompletion(
  messages: Array<{ role: string; content: string }>,
  options: { maxTokens?: number; temperature?: number; jsonMode?: boolean } = {}
): Promise<string> {
  if (!AI_API_KEY) {
    throw new Error('AI_API_KEY or GROQ_API_KEY is required');
  }

  const body: Record<string, unknown> = {
    model: AI_MODEL,
    messages,
    max_tokens: options.maxTokens || 1000,
    temperature: options.temperature || 0.3,
  };

  if (options.jsonMode) {
    body.response_format = { type: 'json_object' };
  }

  const response = await fetch(AI_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${AI_API_KEY}`,
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    throw new Error(`AI API error: ${response.status}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Analyze and translate a news item using AI
 */
export async function analyzeAndTranslateNews(
  news: NewsInput
): Promise<GroqNewsAnalysisResponse | null> {
  try {
    const userPrompt = `
HABER KAYNAĞI: ${news.source}
BAŞLIK: ${news.title}
AÇIKLAMA: ${news.description || 'Açıklama yok'}

Lütfen bu haberi analiz et, Türkçe'ye çevir ve puanla. JSON formatında cevap ver.
`;

    const response = await chatCompletion(
      [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userPrompt },
      ],
      { temperature: 0.3, maxTokens: 1000, jsonMode: true }
    );

    if (!response) {
      console.error('[News AI] No response from analysis engine');
      return null;
    }

    const analysis: GroqNewsAnalysisResponse = JSON.parse(response);

    // Validation
    if (!analysis.impactScore || analysis.impactScore < 1 || analysis.impactScore > 10) {
      console.error('[News AI] Invalid impact score:', analysis.impactScore);
      return null;
    }

    // Filter: Only important news (>= 7)
    if (analysis.impactScore < 7) {
      console.log(`[News AI] Low impact news filtered: ${analysis.impactScore}/10 - ${news.title.substring(0, 50)}...`);
      return null;
    }

    console.log(`[News AI] Important news (${analysis.impactScore}/10): ${analysis.titleTR.substring(0, 60)}...`);
    return analysis;

  } catch (error: any) {
    console.error('[News AI] Error analyzing news:', error.message);
    return null;
  }
}

/**
 * Batch news analysis (rate-limit optimized)
 */
export async function analyzeBatchNews(
  newsArray: NewsInput[],
  maxConcurrent: number = 3
): Promise<GroqNewsAnalysisResponse[]> {
  const results: GroqNewsAnalysisResponse[] = [];

  for (let i = 0; i < newsArray.length; i += maxConcurrent) {
    const chunk = newsArray.slice(i, i + maxConcurrent);

    const chunkResults = await Promise.all(
      chunk.map(news => analyzeAndTranslateNews(news))
    );

    const validResults = chunkResults.filter(r => r !== null) as GroqNewsAnalysisResponse[];
    results.push(...validResults);

    if (i + maxConcurrent < newsArray.length) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  console.log(`[News AI] Analyzed ${newsArray.length} news, ${results.length} passed filter (>= 7/10)`);
  return results;
}

/**
 * Cache key generator
 */
export function generateNewsCacheKey(title: string, source: string): string {
  const normalized = `${title.toLowerCase()}-${source.toLowerCase()}`.replace(/[^a-z0-9]/g, '-');
  return `news:${normalized}:v1`;
}
