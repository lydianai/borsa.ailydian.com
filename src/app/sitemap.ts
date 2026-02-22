import { MetadataRoute } from 'next';

export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
  const currentDate = new Date().toISOString();

  // Ana sayfalar - Yüksek öncelik
  const mainPages = [
    {
      url: baseUrl,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 1.0,
    },
    {
      url: `${baseUrl}/market-scanner`,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 0.95,
    },
    {
      url: `${baseUrl}/trading-signals`,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 0.95,
    },
    {
      url: `${baseUrl}/conservative-signals`,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 0.9,
    },
  ];

  // AI ve Gelişmiş Özellikler - Yüksek öncelik
  const aiPages = [
    {
      url: `${baseUrl}/ai-signals`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/ai-learning-hub`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.85,
    },
    {
      url: `${baseUrl}/ai-learning-hub/rl-agent`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/regime-detection`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/multi-agent`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/meta-learning`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/automl`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/nas`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/explainable-ai`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/federated`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/online-learning`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/ai-learning-hub/causal-ai`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.8,
    },
  ];

  // Quantum ve Premium Özellikler
  const quantumPages = [
    {
      url: `${baseUrl}/quantum-signals`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/quantum-pro`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/quantum-ladder`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.85,
    },
  ];

  // Analiz Araçları
  const analysisPages = [
    {
      url: `${baseUrl}/nirvana`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.85,
    },
    {
      url: `${baseUrl}/alfabetik-pattern`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.85,
    },
    {
      url: `${baseUrl}/bot-analysis`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.85,
    },
    {
      url: `${baseUrl}/charts`,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 0.85,
    },
    {
      url: `${baseUrl}/market-correlation`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/market-insights`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/market-commentary`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/btc-eth-analysis`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/breakout-retest`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
  ];

  // Futures ve Perpetual
  const futuresPages = [
    {
      url: `${baseUrl}/omnipotent-futures`,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 0.9,
    },
    {
      url: `${baseUrl}/perpetual-hub`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.85,
    },
    {
      url: `${baseUrl}/perpetual-hub/whale-tracker`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/perpetual-hub/position-risk`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/perpetual-hub/correlation-matrix`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/perpetual-hub/orderbook-depth`,
      lastModified: currentDate,
      changeFrequency: 'always' as const,
      priority: 0.8,
    },
    {
      url: `${baseUrl}/perpetual-hub/liquidity-flow`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/perpetual-hub/market-microstructure`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/perpetual-hub/sentiment-hedge`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/perpetual-hub/leverage-optimizer`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/perpetual-hub/portfolio-rebalancer`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/perpetual-hub/cross-chain`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/perpetual-hub/contract-scanner`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
  ];

  // Diğer Özellikler
  const otherPages = [
    {
      url: `${baseUrl}/traditional-markets`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.75,
    },
    {
      url: `${baseUrl}/haberler`,
      lastModified: currentDate,
      changeFrequency: 'hourly' as const,
      priority: 0.7,
    },
    {
      url: `${baseUrl}/azure-ai`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.7,
    },
    {
      url: `${baseUrl}/talib`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.7,
    },
    {
      url: `${baseUrl}/auto-trading`,
      lastModified: currentDate,
      changeFrequency: 'daily' as const,
      priority: 0.7,
    },
  ];

  // Statik sayfalar
  const staticPages = [
    {
      url: `${baseUrl}/pricing`,
      lastModified: currentDate,
      changeFrequency: 'weekly' as const,
      priority: 0.6,
    },
    {
      url: `${baseUrl}/settings`,
      lastModified: currentDate,
      changeFrequency: 'monthly' as const,
      priority: 0.3,
    },
    {
      url: `${baseUrl}/login`,
      lastModified: currentDate,
      changeFrequency: 'yearly' as const,
      priority: 0.3,
    },
    {
      url: `${baseUrl}/register`,
      lastModified: currentDate,
      changeFrequency: 'yearly' as const,
      priority: 0.3,
    },
  ];

  return [
    ...mainPages,
    ...aiPages,
    ...quantumPages,
    ...analysisPages,
    ...futuresPages,
    ...otherPages,
    ...staticPages,
  ];
}
