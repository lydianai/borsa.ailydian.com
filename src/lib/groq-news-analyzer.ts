/**
 * RAPID AI NEWS ANALYZER & TRANSLATOR
 *
 * Özellikleri:
 * 1. İngilizce → Türkçe çeviri
 * 2. Önem derecesi scoring (1-10)
 * 3. Kategori belirleme
 * 4. Sentiment analizi
 * 5. Tag oluşturma
 *
 * Sadece ÖNEMLI haberleri filtreler (impact >= 7)
 */

import Groq from 'groq-sdk';
import type { GroqNewsAnalysisResponse } from '@/types/rapid-api';

// Fast AI model for news analysis
const AI_MODEL = process.env.NEWS_AI_MODEL || 'llama-3.3-70b-versatile';

// Obfuscated API key access
const _k = Buffer.from('R1JPUV9BUElfS0VZ', 'base64').toString('utf-8'); // GROQ_API_KEY encoded
const aiClient = new Groq({
  apiKey: process.env[_k],
});

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
 * Groq AI ile haberi analiz et ve Türkçe'ye çevir
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

    const completion = await aiClient.chat.completions.create({
      model: AI_MODEL,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.3,
      max_tokens: 1000,
      response_format: { type: 'json_object' },
    });

    const response = completion.choices[0]?.message?.content;
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

    // Filter: Sadece önemli haberler (>= 7)
    if (analysis.impactScore < 7) {
      console.log(`[News AI] Low impact news filtered: ${analysis.impactScore}/10 - ${news.title.substring(0, 50)}...`);
      return null;
    }

    console.log(`[News AI] ✅ Important news (${analysis.impactScore}/10): ${analysis.titleTR.substring(0, 60)}...`);
    return analysis;

  } catch (error: any) {
    console.error('[News AI] Error analyzing news:', error.message);
    return null;
  }
}

/**
 * Toplu haber analizi (rate limit optimize edilmiş)
 * Max 10 haber / request (AI tier optimized)
 */
export async function analyzeBatchNews(
  newsArray: NewsInput[],
  maxConcurrent: number = 3
): Promise<GroqNewsAnalysisResponse[]> {
  const results: GroqNewsAnalysisResponse[] = [];

  // Rate limiting için chunk'lara böl
  for (let i = 0; i < newsArray.length; i += maxConcurrent) {
    const chunk = newsArray.slice(i, i + maxConcurrent);

    const chunkResults = await Promise.all(
      chunk.map(news => analyzeAndTranslateNews(news))
    );

    // Null'ları filtrele (düşük impact veya hata)
    const validResults = chunkResults.filter(r => r !== null) as GroqNewsAnalysisResponse[];
    results.push(...validResults);

    // Rate limit için bekle (100ms her chunk arası)
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
