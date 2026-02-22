/**
 * Advanced AI Analyzer - LyTrade
 *
 * Provider-agnostic AI market analysis engine.
 * Supports any OpenAI-compatible API (Groq, OpenAI, Ollama, etc.)
 *
 * Configure via environment variables:
 * - AI_API_KEY: Your API key
 * - AI_API_URL: API endpoint (default: https://api.groq.com/openai/v1/chat/completions)
 * - AI_MODEL: Model name (default: llama-3.3-70b-versatile)
 */

// Configuration from environment
const AI_API_URL = process.env.AI_API_URL || 'https://api.groq.com/openai/v1/chat/completions';
const AI_MODEL = process.env.AI_MODEL || 'llama-3.3-70b-versatile';
const AI_API_KEY = process.env.AI_API_KEY || process.env.GROQ_API_KEY || '';

/**
 * Send a chat completion request to any OpenAI-compatible API
 */
async function chatCompletion(
  messages: Array<{ role: string; content: string }>,
  options: { maxTokens?: number; temperature?: number } = {}
): Promise<string> {
  if (!AI_API_KEY) {
    throw new Error(
      'AI_API_KEY (or GROQ_API_KEY) environment variable is required. ' +
      'Get a free key from https://console.groq.com'
    );
  }

  const response = await fetch(AI_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${AI_API_KEY}`,
    },
    body: JSON.stringify({
      model: AI_MODEL,
      messages,
      max_tokens: options.maxTokens || 4096,
      temperature: options.temperature || 0.3,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Unknown error');
    throw new Error(`AI API error (${response.status}): ${errorText}`);
  }

  const data = await response.json();
  return data.choices?.[0]?.message?.content || '';
}

/**
 * Extract JSON from AI response text
 */
function extractJSON(text: string): Record<string, unknown> | null {
  const jsonMatch = text.match(/```json\n([\s\S]+?)\n```/);
  if (jsonMatch && jsonMatch[1]) {
    try {
      return JSON.parse(jsonMatch[1]);
    } catch {
      return null;
    }
  }
  // Try parsing the entire response as JSON
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/**
 * Analyze market data with AI
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
  const prompt = `You are a trading strategy analyzer. Analyze this market data:

${JSON.stringify(marketData, null, 2)}

Tasks:
1. Evaluate the performance of each strategy
2. Suggest the best strategy combination for each coin
3. Suggest parameter improvements
4. Calculate risk scores (0-100) for each coin
5. Return a detailed JSON report

DISCLAIMER: This analysis is for informational purposes only. Not financial advice.

Return your response in this JSON format:

\`\`\`json
{
  "summary": "Overall market summary (1-2 paragraphs)",
  "topStrategies": [
    {
      "name": "Conservative Buy Signal",
      "performance": "excellent|good|moderate|poor",
      "reason": "Why this performance"
    }
  ],
  "coinAnalysis": [
    {
      "symbol": "BTCUSDT",
      "recommendedStrategies": ["Conservative Buy Signal", "Volume Spike"],
      "weights": { "Conservative Buy Signal": 1.5, "Volume Spike": 1.2 },
      "riskScore": 35,
      "riskFactors": ["High volatility", "Strong support level"]
    }
  ],
  "parameterSuggestions": [
    {
      "strategy": "MA Crossover Pullback",
      "parameter": "fastMA",
      "currentValue": 10,
      "suggestedValue": 12,
      "reason": "Fewer false signals"
    }
  ],
  "disclaimer": "This analysis is for informational purposes only. Not financial advice."
}
\`\`\``;

  try {
    const responseText = await chatCompletion(
      [{ role: 'user', content: prompt }],
      { maxTokens: 4096, temperature: 0.3 }
    );

    const parsed = extractJSON(responseText);
    if (parsed) return parsed;

    return { summary: responseText, raw: true };
  } catch (error: any) {
    console.error('Advanced AI Error:', error.message);
    throw new Error(`Advanced AI analysis failed: ${error.message}`);
  }
}

/**
 * Evaluate strategy performance with AI
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
  const prompt = `Evaluate this trading strategy:

**Strategy**: ${strategyData.strategyName}
**Success Rate**: ${strategyData.successRate}%
**Avg Confidence**: ${strategyData.avgConfidence}%

**Recent Signals**:
${JSON.stringify(strategyData.recentSignals, null, 2)}

Return JSON:
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
    const responseText = await chatCompletion(
      [{ role: 'user', content: prompt }],
      { maxTokens: 2048, temperature: 0.3 }
    );

    const parsed = extractJSON(responseText);
    if (parsed) return parsed;

    return { raw: responseText };
  } catch (error: any) {
    console.error('Strategy Evaluation Error:', error.message);
    throw error;
  }
}

/**
 * Suggest adaptive coin weights using AI
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
  const prompt = `Suggest strategy weights for ${coinData.symbol}:

**Historical Performance**:
${JSON.stringify(coinData.historicalPerformance, null, 2)}

**Current Market**:
- Volatility: ${coinData.currentMarket.volatility}%
- 24h Volume: $${coinData.currentMarket.volume24h.toLocaleString()}
- Trend: ${coinData.currentMarket.trend}

Return weights (0.5-2.0) for each strategy as JSON:
\`\`\`json
{
  "weights": { "Strategy Name": 1.5 },
  "reasoning": { "Strategy Name": "Reason for weight" }
}
\`\`\``;

  try {
    const responseText = await chatCompletion(
      [{ role: 'user', content: prompt }],
      { maxTokens: 1536, temperature: 0.2 }
    );

    const parsed = extractJSON(responseText);
    if (parsed) return parsed;

    return { weights: {}, reasoning: {} };
  } catch (error: any) {
    console.error('Weight Suggestion Error:', error.message);
    throw error;
  }
}

/**
 * Health check for AI service
 */
export async function checkAdvancedAIHealth() {
  try {
    const responseText = await chatCompletion(
      [{ role: 'user', content: 'Test: Respond with "OK"' }],
      { maxTokens: 50 }
    );

    return {
      status: 'healthy',
      model: AI_MODEL,
      provider: new URL(AI_API_URL).hostname,
      response: responseText.trim(),
    };
  } catch (error: any) {
    return {
      status: 'unhealthy',
      error: error.message,
    };
  }
}

export default {
  analyzeMarketWithAdvancedAI,
  evaluateStrategyPerformance,
  suggestCoinWeights,
  checkAdvancedAIHealth,
};
