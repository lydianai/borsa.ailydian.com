'use client';

/**
 * Pricing Page - Subscription Tiers
 *
 * White-hat compliance: Transparent pricing for legitimate trading signal service
 * Educational and analytical purposes only
 */

import { useState } from 'react';
import { Check, Zap, TrendingUp, Building2 } from 'lucide-react';
import { getAllTiers, type SubscriptionTier } from '@/lib/stripe/config';

const tierIcons = {
  free: TrendingUp,
  starter: Zap,
  pro: TrendingUp,
  enterprise: Building2,
};

const tierColors = {
  free: 'from-gray-600 to-gray-700',
  starter: 'from-blue-600 to-blue-700',
  pro: 'from-purple-600 to-purple-700',
  enterprise: 'from-amber-600 to-amber-700',
};

export default function PricingPage() {
  const [billingCycle, setBillingCycle] = useState<'monthly' | 'annual'>('monthly');
  const tiers = getAllTiers();

  const handleSubscribe = async (tier: SubscriptionTier) => {
    if (tier === 'free') {
      // Redirect to signup
      window.location.href = '/signup';
      return;
    }

    try {
      // TODO: Call checkout API
      const response = await fetch('/api/stripe/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tier, billingCycle }),
      });

      const { url } = await response.json();
      window.location.href = url;
    } catch (error) {
      console.error('Checkout error:', error);
      alert('Something went wrong. Please try again.');
    }
  };

  const getPrice = (basePrice: number) => {
    if (billingCycle === 'annual') {
      // 20% discount for annual
      return Math.floor(basePrice * 12 * 0.8);
    }
    return basePrice;
  };

  return (
    <div className="min-h-screen bg-black text-white py-12 px-4" style={{ paddingTop: '80px' }}>
      {/* Header */}
      <div className="max-w-7xl mx-auto text-center mb-12">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 text-transparent bg-clip-text">
          Planınızı Seçin
        </h1>
        <p className="text-xl text-gray-400 mb-8">
          14 gün ücretsiz deneme ile başlayın. Kredi kartı gerekmez.
        </p>

        {/* Billing Toggle */}
        <div className="inline-flex items-center gap-4 bg-gray-900 rounded-full p-1">
          <button
            onClick={() => setBillingCycle('monthly')}
            className={`px-6 py-2 rounded-full transition-all ${
              billingCycle === 'monthly'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Aylık
          </button>
          <button
            onClick={() => setBillingCycle('annual')}
            className={`px-6 py-2 rounded-full transition-all ${
              billingCycle === 'annual'
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            Yıllık
            <span className="ml-2 text-xs bg-green-500 text-white px-2 py-1 rounded-full">
              %20 İndirim
            </span>
          </button>
        </div>
      </div>

      {/* Pricing Cards */}
      <div className="max-w-7xl mx-auto grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {tiers.map((tier) => {
          const Icon = tierIcons[tier.id as keyof typeof tierIcons];
          const isPopular = 'popular' in tier && tier.popular;
          const displayPrice = getPrice(tier.price);

          return (
            <div
              key={tier.id}
              className={`relative bg-gray-900 rounded-2xl border-2 ${
                isPopular ? 'border-purple-500' : 'border-gray-800'
              } p-6 hover:border-blue-500 transition-all`}
            >
              {/* Popular Badge */}
              {isPopular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-4 py-1 rounded-full text-sm font-bold">
                    EN POPÜLER
                  </div>
                </div>
              )}

              {/* Icon & Name */}
              <div className="mb-6">
                <div
                  className={`w-12 h-12 rounded-xl bg-gradient-to-br ${
                    tierColors[tier.id as keyof typeof tierColors]
                  } flex items-center justify-center mb-4`}
                >
                  <Icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-2xl font-bold">{tier.name}</h3>
              </div>

              {/* Price */}
              <div className="mb-6">
                {tier.price === 0 ? (
                  <div>
                    <span className="text-4xl font-bold">Ücretsiz</span>
                  </div>
                ) : (
                  <div>
                    <span className="text-4xl font-bold">
                      ${billingCycle === 'annual' ? displayPrice : tier.price}
                    </span>
                    <span className="text-gray-400 ml-2">
                      /{billingCycle === 'annual' ? 'yıl' : 'ay'}
                    </span>
                    {billingCycle === 'annual' && (
                      <div className="text-sm text-green-500 mt-1">
                        Yıllık ${tier.price}/ay
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Features */}
              <ul className="space-y-3 mb-8">
                {tier.features.map((feature, idx) => (
                  <li key={idx} className="flex items-start gap-3">
                    <Check className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                    <span className="text-sm text-gray-300">{feature}</span>
                  </li>
                ))}
              </ul>

              {/* CTA Button */}
              <button
                onClick={() => handleSubscribe(tier.id)}
                className={`w-full py-3 rounded-xl font-bold transition-all ${
                  isPopular
                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
                    : tier.id === 'free'
                    ? 'bg-gray-800 hover:bg-gray-700'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {tier.id === 'free' ? 'Başla' : 'Ücretsiz Deneyin'}
              </button>

              {tier.id !== 'free' && (
                <p className="text-xs text-gray-500 text-center mt-3">
                  14 gün ücretsiz, sonra ${tier.price}/ay
                </p>
              )}
            </div>
          );
        })}
      </div>

      {/* FAQ Section */}
      <div className="max-w-4xl mx-auto mt-20">
        <h2 className="text-3xl font-bold text-center mb-12">
          Sıkça Sorulan Sorular
        </h2>

        <div className="space-y-6">
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-lg font-bold mb-2">
              İstediğim zaman iptal edebilir miyim?
            </h3>
            <p className="text-gray-400">
              Evet, aboneliğinizi istediğiniz zaman iptal edebilirsiniz. Erişiminiz fatura döneminin sonuna kadar devam eder.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-lg font-bold mb-2">
              Hangi ödeme yöntemlerini kabul ediyorsunuz?
            </h3>
            <p className="text-gray-400">
              Tüm büyük kredi kartlarını (Visa, MasterCard, American Express) ve Stripe üzerinden diğer ödeme yöntemlerini kabul ediyoruz.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-lg font-bold mb-2">
              Ücretsiz deneme var mı?
            </h3>
            <p className="text-gray-400">
              Evet! Tüm ücretli planlar 14 günlük ücretsiz deneme içerir. Başlamak için kredi kartı gerekmez.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-lg font-bold mb-2">
              Yükseltme veya düşürme yapabilir miyim?
            </h3>
            <p className="text-gray-400">
              Evet, planınızı istediğiniz zaman değiştirebilirsiniz. Yükseltmeler hemen geçerli olur, düşürmeler ise fatura döneminin sonunda aktif olur.
            </p>
          </div>

          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h3 className="text-lg font-bold mb-2">
              Sinyaller gerçek zamanlı mı?
            </h3>
            <p className="text-gray-400">
              Evet! Tüm ücretli planlar AI modellerimizden ve teknik analizden gerçek zamanlı sinyaller alır. Ücretsiz planda eğitim amaçlı 24 saat gecikme vardır.
            </p>
          </div>
        </div>
      </div>

      {/* Trust Badges */}
      <div className="max-w-7xl mx-auto mt-20 text-center">
        <p className="text-gray-500 text-sm mb-4">Dünya çapında yatırımcılar tarafından güvenilir</p>
        <div className="flex justify-center items-center gap-8 flex-wrap">
          <div className="text-gray-600">
            <div className="text-2xl font-bold text-white">99.9%</div>
            <div className="text-sm">Çalışma Süresi</div>
          </div>
          <div className="text-gray-600">
            <div className="text-2xl font-bold text-white">500+</div>
            <div className="text-sm">İndikatör</div>
          </div>
          <div className="text-gray-600">
            <div className="text-2xl font-bold text-white">100+</div>
            <div className="text-sm">AI Modeli</div>
          </div>
          <div className="text-gray-600">
            <div className="text-2xl font-bold text-white">24/7</div>
            <div className="text-sm">Destek</div>
          </div>
        </div>
      </div>

      {/* Compliance Notice */}
      <div className="max-w-4xl mx-auto mt-12 text-center text-xs text-gray-600 border-t border-gray-900 pt-6">
        <p>
          ⚠️ <strong>Eğitim ve Analiz Aracı:</strong> LyTrade Scanner sadece eğitim ve analiz amaçlı
          ticaret sinyalleri sağlar. Tüm ticaret risk içerir. Geçmiş performans gelecekteki sonuçları
          garanti etmez. Lütfen sorumlu bir şekilde ticaret yapın ve kaybetmeyi göze alamayacağınız
          miktardan fazla yatırım yapmayın.
        </p>
      </div>
    </div>
  );
}
