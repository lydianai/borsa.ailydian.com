/**
 * AI CHAT ASSISTANT API
 * Trading analysis and recommendations
 */

import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

// Lightweight AI responses (replace with actual ML model or external API)
function generateAIResponse(userMessage: string): string {
  const msg = userMessage.toLowerCase();

  // Crypto patterns
  if (msg.includes('btc') || msg.includes('bitcoin')) {
    return `📊 **Bitcoin (BTC) Analizi:**

**Teknik Göstergeler:**
• Fiyat: $117,782 (+4.15% 24h)
• RSI: 68 (Nötr bölge)
• MACD: Pozitif momentum
• Support: $112,000
• Resistance: $120,000

**AI Değerlendirmesi:**
Bitcoin güçlü bir yükseliş trendinde. Kısa vadede $120,000 direnci test edilebilir. Orta vadeli görünüm pozitif.

**Öneri:** HOLD (Uzun vadeli yatırımcılar için) veya dikkatli AL pozisyonu (kısa vadeli)

⚠️ Risk: Orta | Güven: %78`;
  }

  if (msg.includes('eth') || msg.includes('ethereum')) {
    return `📊 **Ethereum (ETH) Analizi:**

**Teknik Göstergeler:**
• Fiyat: $4,334 (+5.22% 24h)
• RSI: 72 (Aşırı alım bölgesine yakın)
• MACD: Güçlü pozitif
• Support: $4,100
• Resistance: $4,500

**AI Değerlendirmesi:**
Ethereum Bitcoin'den daha güçlü performans gösteriyor. Layer-2 gelişmeleri fiyatı destekliyor.

**Öneri:** AL (Düzeltmelerde) - $4,200 altı iyi giriş noktası

⚠️ Risk: Orta-Yüksek | Güven: %82`;
  }

  if (msg.includes('xrp') || msg.includes('ripple')) {
    return `📊 **XRP Analizi:**

**Teknik Göstergeler:**
• Fiyat: $2.94 (+4.03% 24h)
• RSI: 58 (Nötr)
• Hacim: Yüksek
• Support: $2.80
• Resistance: $3.00

**AI Değerlendirmesi:**
XRP yasal belirsizlikler azaldıkça güçleniyor. $3.00 direnci kritik seviye.

**Öneri:** BEKLE - $3.00 kırılırsa AL sinyali

⚠️ Risk: Yüksek | Güven: %65`;
  }

  // Stocks
  if (msg.includes('aapl') || msg.includes('apple')) {
    return `📊 **Apple (AAPL) Analizi:**

**Temel Veriler:**
• P/E Ratio: 28.5
• Piyasa Değeri: $3.2T
• Temettü: %0.52

**Teknik Durum:**
• 50-day MA: Üzerinde
• 200-day MA: Üzerinde
• Trend: Yükseliş

**AI Değerlendirmesi:**
Apple güçlü fundamentals ve teknik görünüme sahip. Vision Pro ve AI girişimleri destekleyici.

**Öneri:** AL (Uzun vadeli portföy için)

⚠️ Risk: Düşük | Güven: %85`;
  }

  // Genel stratejiler
  if (msg.includes('strateji') || msg.includes('nasıl')) {
    return `🎯 **Genel Trading Stratejileri:**

**1. DCA (Dollar Cost Averaging)**
• Düzenli aralıklarla sabit miktarda al
• Risk: Düşük
• Uygun: Yeni başlayanlar

**2. Trend Following**
• Güçlü trendleri takip et
• Risk: Orta
• Uygun: Deneyimli traderlar

**3. Mean Reversion**
• Aşırı satım/alımda pozisyon al
• Risk: Orta-Yüksek
• Uygun: Teknik analiz bilenler

**4. Position Trading**
• Uzun vadeli tutma
• Risk: Düşük
• Uygun: Pasif yatırımcılar

💡 Hangi strateji hakkında detay istersiniz?`;
  }

  if (msg.includes('risk')) {
    return `🛡️ **Risk Yönetimi Prensipleri:**

**1. Pozisyon Boyutlandırma**
• Her işlemde sermayenin %1-2'sinden fazlasını riske atma

**2. Stop Loss Kullan**
• Her pozisyonda mutlaka stop loss belirle
• Zarar kesme disiplini

**3. Diversifikasyon**
• Portföyünü farklı varlıklara yay
• Tek varlığa bağımlı kalma

**4. Risk/Reward Oranı**
• Minimum 1:2 (1 birim risk, 2 birim kazanç)
• İdeal 1:3 veya daha yüksek

**5. Duygusal Disiplin**
• Planına sadık kal
• FOMO'dan kaçın

⚠️ Risk yönetimi uzun vadeli başarının anahtarıdır!`;
  }

  // Portföy
  if (msg.includes('portföy')) {
    return `💼 **Dengeli Portföy Önerisi:**

**Muhafazakar (Düşük Risk)**
• 60% BTC
• 30% ETH
• 10% Stablecoin
Beklenen Getiri: %15-25/yıl

**Dengeli (Orta Risk)**
• 40% BTC
• 30% ETH
• 20% Top 10 Altcoinler
• 10% Stablecoin
Beklenen Getiri: %30-50/yıl

**Agresif (Yüksek Risk)**
• 30% BTC
• 30% ETH
• 30% Altcoinler
• 10% Yeni Projeler
Beklenen Getiri: %50-100+/yıl

💡 Risk toleransınıza göre seçim yapın!`;
  }

  // Default response
  return `🤖 **Anlıyorum!**

Size yardımcı olmak isterim. Şu konularda sorular sorabilirsiniz:

📈 **Kripto Analiz:**
• BTC, ETH, XRP ve diğer coinler hakkında

📊 **Hisse Senedi Analiz:**
• AAPL, TSLA, MSFT ve diğer hisseler

🎯 **Strateji:**
• Trading stratejileri
• Risk yönetimi
• Portföy oluşturma

💡 **İpucu:** "BTC analiz et", "ETH al mı satmalı mı?", "risk yönetimi" gibi sorular sorun!

Hangi konuda yardım istersiniz?`;
}

export async function POST(request: NextRequest) {
  try {
    const { message } = await request.json();

    if (!message || typeof message !== 'string') {
      return NextResponse.json({
        success: false,
        error: 'Message is required'
      }, { status: 400 });
    }

    // Generate AI response
    const response = generateAIResponse(message);

    return NextResponse.json({
      success: true,
      response,
      timestamp: new Date().toISOString()
    });

  } catch (error: any) {
    console.error('AI Chat error:', error);
    return NextResponse.json({
      success: false,
      error: error.message
    }, { status: 500 });
  }
}
