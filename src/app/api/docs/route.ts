/**
 * API Documentation Endpoint
 *
 * Provides OpenAPI/Swagger documentation for all API endpoints
 */

import { NextResponse } from 'next/server';

const API_DOCS = {
  openapi: '3.0.0',
  info: {
    title: 'LyTrade Scanner API',
    version: '1.0.0',
    description: 'AI-Powered Cryptocurrency Trading Signal Platform API',
    contact: {
      name: 'LyTrade Support',
      url: 'https://github.com/AiLydian/lytrade-scanner',
    },
  },
  servers: [
    {
      url: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
      description: 'Local development server',
    },
  ],
  paths: {
    '/api/health': {
      get: {
        summary: 'Health Check',
        description: 'Check if the API is running and healthy',
        tags: ['System'],
        responses: {
          '200': {
            description: 'API is healthy',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    status: { type: 'string', example: 'ok' },
                    message: { type: 'string', example: 'Backend API is running' },
                    timestamp: { type: 'string', format: 'date-time' },
                    version: { type: 'string', example: '2.0.0-backend-only' },
                  },
                },
              },
            },
          },
        },
      },
    },
    '/api/binance/futures': {
      get: {
        summary: 'Get Binance Futures Market Data',
        description: 'Fetch real-time market data for all USDT Perpetual Futures (617 markets)',
        tags: ['Market Data'],
        responses: {
          '200': {
            description: 'Market data retrieved successfully',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    success: { type: 'boolean' },
                    data: {
                      type: 'object',
                      properties: {
                        all: { type: 'array', description: 'All 617 markets' },
                        topVolume: { type: 'array', description: 'Top 20 by volume' },
                        topGainers: { type: 'array', description: 'Top 10 gainers' },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
    '/api/conservative-signals': {
      get: {
        summary: 'Get Conservative Buy Signals',
        description: 'Ultra-safe buy signals with 4/5 conditions met (80-95% confidence)',
        tags: ['Signals'],
        responses: {
          '200': {
            description: 'Signals generated successfully',
            content: {
              'application/json': {
                schema: {
                  type: 'object',
                  properties: {
                    success: { type: 'boolean' },
                    data: {
                      type: 'object',
                      properties: {
                        signals: { type: 'array' },
                        stats: {
                          type: 'object',
                          properties: {
                            total: { type: 'number' },
                            avgConfidence: { type: 'number' },
                          },
                        },
                      },
                    },
                  },
                },
              },
            },
          },
        },
      },
    },
    '/api/breakout-retest': {
      get: {
        summary: 'Get Breakout-Retest Signals',
        description: '3-phase pattern recognition (Consolidation → Breakout → Retest)',
        tags: ['Signals'],
        responses: {
          '200': {
            description: 'Breakout-retest patterns found',
          },
        },
      },
    },
    '/api/ai-signals': {
      get: {
        summary: 'Get AI Deep Analysis Signals',
        description: 'Groq AI-powered deep market analysis in Turkish',
        tags: ['AI Signals'],
        responses: {
          '200': {
            description: 'AI analysis completed',
          },
        },
      },
    },
  },
  tags: [
    { name: 'System', description: 'System health and status endpoints' },
    { name: 'Market Data', description: 'Real-time market data from exchanges' },
    { name: 'Signals', description: 'Trading signal generation' },
    { name: 'AI Signals', description: 'AI-powered signal analysis' },
  ],
  components: {
    schemas: {
      Error: {
        type: 'object',
        properties: {
          success: { type: 'boolean', example: false },
          error: { type: 'string' },
          message: { type: 'string' },
        },
      },
    },
  },
};

export async function GET() {
  return NextResponse.json(API_DOCS, {
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'public, max-age=3600',
    },
  });
}
