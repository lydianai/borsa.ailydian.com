import { NextRequest, NextResponse } from 'next/server';

/**
 * Azure OpenAI Market Analysis Endpoint
 * Provides AI-powered market insights for futures trading
 */

interface MarketAnalysisRequest {
  symbol: string;
  timeframe: string;
  marketData?: {
    price: number;
    volume: number;
    change24h: number;
  };
}

export async function POST(request: NextRequest) {
  try {
    const body: MarketAnalysisRequest = await request.json();
    const { symbol, timeframe, marketData } = body;

    // Azure OpenAI endpoint (fallback to local if not configured)
    const azureEndpoint = process.env.AZURE_OPENAI_ENDPOINT;
    const azureApiKey = process.env.AZURE_OPENAI_API_KEY;
    const deploymentName = process.env.AZURE_OPENAI_DEPLOYMENT_NAME || 'gpt-4';

    if (!azureEndpoint || !azureApiKey) {
      // Fallback to mock analysis if Azure not configured
      return NextResponse.json({
        success: true,
        analysis: {
          summary: `${symbol} analysis for ${timeframe} timeframe`,
          sentiment: 'NEUTRAL',
          confidence: 0.75,
          keyPoints: [
            'Market showing consolidation pattern',
            'Volume analysis suggests accumulation',
            'Technical indicators neutral',
          ],
          recommendation: 'HOLD',
          riskLevel: 'MEDIUM',
        },
        source: 'local-fallback',
      });
    }

    // Call Azure OpenAI for market analysis
    const prompt = `Analyze the cryptocurrency market for ${symbol} on ${timeframe} timeframe.
${marketData ? `Current Price: $${marketData.price}
24h Change: ${marketData.change24h}%
Volume: ${marketData.volume}` : ''}

Provide a concise market analysis including:
1. Overall market sentiment
2. Key technical indicators
3. Risk assessment
4. Trading recommendation

Format as JSON with keys: sentiment, confidence, keyPoints (array), recommendation, riskLevel`;

    const azureResponse = await fetch(
      `${azureEndpoint}/openai/deployments/${deploymentName}/chat/completions?api-version=2024-02-15-preview`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'api-key': azureApiKey,
        },
        body: JSON.stringify({
          messages: [
            {
              role: 'system',
              content: 'You are an expert cryptocurrency market analyst and trading advisor.',
            },
            {
              role: 'user',
              content: prompt,
            },
          ],
          max_tokens: 500,
          temperature: 0.7,
        }),
      }
    );

    if (!azureResponse.ok) {
      throw new Error('Azure OpenAI request failed');
    }

    const azureData = await azureResponse.json();
    const analysisText = azureData.choices[0].message.content;

    // Try to parse JSON from response
    let analysis;
    try {
      analysis = JSON.parse(analysisText);
    } catch {
      // If not JSON, create structured response from text
      analysis = {
        summary: analysisText,
        sentiment: 'NEUTRAL',
        confidence: 0.75,
        keyPoints: analysisText.split('\n').filter((line: string) => line.trim()),
        recommendation: 'HOLD',
        riskLevel: 'MEDIUM',
      };
    }

    return NextResponse.json({
      success: true,
      analysis,
      source: 'azure-openai',
      model: deploymentName,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('Azure market analysis error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Market analysis failed',
      },
      { status: 500 }
    );
  }
}
