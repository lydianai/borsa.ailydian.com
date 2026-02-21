import { NextRequest, NextResponse } from 'next/server';

const AI_LEARNING_URL = process.env.AI_LEARNING_URL || 'http://localhost:5020';

export async function GET() {
  try {
    const response = await fetch(`${AI_LEARNING_URL}/system/stats`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch system stats');
    }

    const data = await response.json();

    return NextResponse.json(data);
  } catch (error: any) {
    console.error('❌ AI System Stats Error:', error);

    // Return mock data if service unavailable
    return NextResponse.json({
      success: true,
      mock: true,
      timestamp: new Date().toISOString(),
      rl_agent: {
        episodes: 12847,
        win_rate: 73.2,
        learning_rate: 98.5,
      },
      online_learning: {
        updates: 2458,
        accuracy: 91.3,
        drift_score: 0.12,
      },
      multi_agent: {
        agents: 5,
        best_agent: 'momentum',
        ensemble_acc: 94.7,
      },
      automl: {
        trials: 1247,
        best_sharpe: 2.84,
      },
      nas: {
        generations: 248,
        best_arch: 'Transformer',
      },
      meta_learning: {
        adaptation: 96.2,
        transfer: 85.0,
      },
      federated: {
        users: 8247,
        global_acc: 93.1,
      },
      causal: {
        paths: 247,
        confidence: 87.5,
      },
      regime: {
        current: 'Bull',
        confidence: 92.3,
      },
      explainable: {
        explainability: 96.8,
        top_feature: 'Volume',
      },
    });
  }
}
