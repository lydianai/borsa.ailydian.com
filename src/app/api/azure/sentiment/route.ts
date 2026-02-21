import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { text } = await request.json();

    if (!text) {
      return NextResponse.json(
        { success: false, error: 'Text is required' },
        { status: 400 }
      );
    }

    // Check if Azure Text Analytics is configured
    const azureEndpoint = process.env.AZURE_TEXT_ANALYTICS_ENDPOINT;
    const azureApiKey = process.env.AZURE_TEXT_ANALYTICS_KEY;

    if (!azureEndpoint || !azureApiKey) {
      // Fallback: Simple keyword-based sentiment
      const sentiment = analyzeSimpleSentiment(text);
      return NextResponse.json({
        success: true,
        sentiment,
        source: 'local-fallback',
      });
    }

    // Call Azure Text Analytics
    const response = await fetch(`${azureEndpoint}/text/analytics/v3.1/sentiment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': azureApiKey,
      },
      body: JSON.stringify({
        documents: [{ id: '1', language: 'en', text }],
      }),
    });

    if (!response.ok) {
      throw new Error('Azure Text Analytics failed');
    }

    const data = await response.json();
    const sentiment = data.documents[0]?.sentiment || 'neutral';

    return NextResponse.json({
      success: true,
      sentiment: sentiment.toUpperCase(),
      scores: data.documents[0]?.confidenceScores,
      source: 'azure',
    });
  } catch (error: any) {
    console.error('Sentiment analysis error:', error);

    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Sentiment analysis failed',
      },
      { status: 500 }
    );
  }
}

// Simple fallback sentiment analysis
function analyzeSimpleSentiment(text: string): 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' {
  const lowerText = text.toLowerCase();

  const positiveWords = ['buy', 'bullish', 'moon', 'profit', 'gain', 'up', 'rise', 'surge', 'rally'];
  const negativeWords = ['sell', 'bearish', 'dump', 'loss', 'down', 'fall', 'crash', 'dip'];

  let score = 0;

  positiveWords.forEach(word => {
    if (lowerText.includes(word)) score++;
  });

  negativeWords.forEach(word => {
    if (lowerText.includes(word)) score--;
  });

  if (score > 0) return 'POSITIVE';
  if (score < 0) return 'NEGATIVE';
  return 'NEUTRAL';
}
