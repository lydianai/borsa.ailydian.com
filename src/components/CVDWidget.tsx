'use client';

import { Icons } from '@/components/Icons';

interface CVDWidgetProps {
  cvdData: any;
}

export function CVDWidget({ cvdData }: CVDWidgetProps) {
  if (!cvdData) return null;

  return (
    <div style={{
      background: 'linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(124, 58, 237, 0.1) 100%)',
      backdropFilter: 'blur(20px)',
      border: '2px solid rgba(139, 92, 246, 0.4)',
      borderRadius: '16px',
      padding: '32px',
      marginTop: '24px',
      boxShadow: '0 8px 32px rgba(139, 92, 246, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.1)',
      transition: 'all 0.3s ease'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '32px' }}>
        <div style={{
          width: '56px',
          height: '56px',
          background: 'linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%)',
          borderRadius: '14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 6px 20px rgba(139, 92, 246, 0.4)'
        }}>
          <Icons.TrendingUp style={{ width: '30px', height: '30px', color: '#FFFFFF' }} />
        </div>
        <div>
          <h2 style={{ fontSize: '26px', fontWeight: '700', color: '#FFFFFF', marginBottom: '4px' }}>
            CVD - Kurumsal Trader'ların Gizli Silahı
          </h2>
          <p style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.6)' }}>
            Büyük oyuncuların gerçek alım/satım hareketlerini anında görün - Fiyat grafiğinin gösteremediği sırları keşfedin
          </p>
        </div>
      </div>

      {/* Divergence Alert at Top */}
      {cvdData.divergence && cvdData.divergence.type !== 'NEUTRAL' && (
        <div style={{
          padding: '20px',
          background: cvdData.divergence.type === 'BEARISH'
            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.15) 100%)'
            : 'linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%)',
          border: `2px solid ${cvdData.divergence.type === 'BEARISH' ? 'rgba(239, 68, 68, 0.5)' : 'rgba(16, 185, 129, 0.5)'}`,
          borderRadius: '12px',
          marginBottom: '24px'
        }}>
          <div style={{ fontSize: '14px', fontWeight: '700', color: cvdData.divergence.type === 'BEARISH' ? '#EF4444' : '#10B981', marginBottom: '8px' }}>
            {cvdData.divergence.type === 'BEARISH' ? '⚠️ BEARISH DIVERGENCE' : '🔥 BULLISH DIVERGENCE'}
          </div>
          <div style={{ fontSize: '16px', color: '#FFFFFF', lineHeight: '1.6' }}>
            {cvdData.divergence.message}
          </div>
          <div style={{ marginTop: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.7)' }}>
              Sinyal Gücü:
            </div>
            <div style={{
              width: '200px',
              height: '8px',
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${cvdData.divergence.strength}%`,
                height: '100%',
                background: cvdData.divergence.type === 'BEARISH' ? '#EF4444' : '#10B981',
                transition: 'width 0.5s ease'
              }} />
            </div>
            <div style={{ fontSize: '14px', fontWeight: '700', color: cvdData.divergence.type === 'BEARISH' ? '#EF4444' : '#10B981' }}>
              {cvdData.divergence.strength.toFixed(0)}%
            </div>
          </div>
        </div>
      )}

      {/* Main Metrics Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '32px' }}>
        {/* Current CVD */}
        <div style={{
          padding: '20px',
          background: 'rgba(139, 92, 246, 0.1)',
          border: '1px solid rgba(139, 92, 246, 0.3)',
          borderRadius: '12px'
        }}>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
            GÜNCEL CVD DEĞERİ
          </div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: cvdData.currentCVD > 0 ? '#10B981' : '#EF4444' }}>
            {cvdData.currentCVD > 0 ? '+' : ''}{cvdData.currentCVD.toFixed(2)}
          </div>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
            {cvdData.cvdTrend === 'RISING' ? '📈 Alıcılar güçleniyor' : cvdData.cvdTrend === 'FALLING' ? '📉 Satıcılar baskın' : '➡️ Denge durumu'}
          </div>
          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginTop: '6px', fontStyle: 'italic' }}>
            {cvdData.currentCVD > 0 ? 'Pozitif = Kurumsal alım devam ediyor' : 'Negatif = Kurumsal satış baskısı var'}
          </div>
        </div>

        {/* Buy Pressure */}
        <div style={{
          padding: '20px',
          background: 'rgba(16, 185, 129, 0.1)',
          border: '1px solid rgba(16, 185, 129, 0.3)',
          borderRadius: '12px'
        }}>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
            ALIŞ BASKISI (BUY PRESSURE)
          </div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: '#10B981' }}>
            {cvdData.buyPressure.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
            Toplam Alış: {cvdData.totalBuyVolume.toFixed(0)} BTC
          </div>
          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginTop: '6px', fontStyle: 'italic' }}>
            {cvdData.buyPressure > 55 ? 'Güçlü alım = Yükseliş potansiyeli' : cvdData.buyPressure < 45 ? 'Zayıf alım = Dikkatli ol' : 'Dengeli durum'}
          </div>
        </div>

        {/* Sell Pressure */}
        <div style={{
          padding: '20px',
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid rgba(239, 68, 68, 0.3)',
          borderRadius: '12px'
        }}>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
            SATIŞ BASKISI (SELL PRESSURE)
          </div>
          <div style={{ fontSize: '24px', fontWeight: '700', color: '#EF4444' }}>
            {cvdData.sellPressure.toFixed(1)}%
          </div>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
            Toplam Satış: {cvdData.totalSellVolume.toFixed(0)} BTC
          </div>
          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginTop: '6px', fontStyle: 'italic' }}>
            {cvdData.sellPressure > 55 ? 'Yüksek satış = Düşüş riski' : cvdData.sellPressure < 45 ? 'Düşük satış = İyi sinyal' : 'Dengeli durum'}
          </div>
        </div>

        {/* Dominant Force */}
        <div style={{
          padding: '20px',
          background: cvdData.dominantForce === 'WHALE'
            ? 'rgba(234, 179, 8, 0.1)'
            : cvdData.dominantForce === 'INSTITUTIONAL'
            ? 'rgba(59, 130, 246, 0.1)'
            : 'rgba(107, 114, 128, 0.1)',
          border: `1px solid ${cvdData.dominantForce === 'WHALE'
            ? 'rgba(234, 179, 8, 0.3)'
            : cvdData.dominantForce === 'INSTITUTIONAL'
            ? 'rgba(59, 130, 246, 0.3)'
            : 'rgba(107, 114, 128, 0.3)'}`,
          borderRadius: '12px'
        }}>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '8px', fontWeight: '600' }}>
            PİYASAYI KİM KONTROL EDİYOR?
          </div>
          <div style={{ fontSize: '18px', fontWeight: '700', color: cvdData.dominantForce === 'WHALE' ? '#EAB308' : cvdData.dominantForce === 'INSTITUTIONAL' ? '#3B82F6' : '#6B7280' }}>
            {cvdData.dominantForce === 'WHALE' ? '🐋 BALINALAR' : cvdData.dominantForce === 'INSTITUTIONAL' ? '🏛️ KURUMSAL' : '🐟 RETAIL'}
          </div>
          <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)', marginTop: '4px' }}>
            {cvdData.dominantForce === 'WHALE' ? 'Büyük para hareketi var' : cvdData.dominantForce === 'INSTITUTIONAL' ? 'Profesyoneller aktif' : 'Küçük yatırımcılar aktif'}
          </div>
          <div style={{ fontSize: '10px', color: 'rgba(255, 255, 255, 0.4)', marginTop: '6px', fontStyle: 'italic' }}>
            {cvdData.dominantForce === 'WHALE' ? '>$1M işlemler baskın' : cvdData.dominantForce === 'INSTITUTIONAL' ? '$100k-$1M işlemler baskın' : '<$100k işlemler baskın'}
          </div>
        </div>
      </div>

      {/* Size Analysis Breakdown */}
      <div style={{
        background: 'rgba(0, 0, 0, 0.3)',
        borderRadius: '12px',
        padding: '24px',
        marginBottom: '24px'
      }}>
        <h3 style={{ fontSize: '16px', fontWeight: '700', color: '#FFFFFF', marginBottom: '20px' }}>
          📊 Büyüklük Analizi - Kim Alıyor, Kim Satıyor?
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          {/* Retail */}
          <div>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px', fontWeight: '600' }}>
              🐟 RETAIL (&lt;$100k)
            </div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <div style={{ flex: 1, padding: '10px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '6px' }}>
                <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '2px' }}>Alış</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981' }}>{cvdData.sizeAnalysis.retail.buyVolume.toFixed(0)}</div>
              </div>
              <div style={{ flex: 1, padding: '10px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '6px' }}>
                <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '2px' }}>Satış</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444' }}>{cvdData.sizeAnalysis.retail.sellVolume.toFixed(0)}</div>
              </div>
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
              Oran: {cvdData.sizeAnalysis.retail.percentage.toFixed(1)}%
            </div>
          </div>

          {/* Institutional */}
          <div>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px', fontWeight: '600' }}>
              🏛️ KURUMSAL ($100k-$1M)
            </div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <div style={{ flex: 1, padding: '10px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '6px' }}>
                <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '2px' }}>Alış</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981' }}>{cvdData.sizeAnalysis.institutional.buyVolume.toFixed(0)}</div>
              </div>
              <div style={{ flex: 1, padding: '10px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '6px' }}>
                <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '2px' }}>Satış</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444' }}>{cvdData.sizeAnalysis.institutional.sellVolume.toFixed(0)}</div>
              </div>
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
              Oran: {cvdData.sizeAnalysis.institutional.percentage.toFixed(1)}%
            </div>
          </div>

          {/* Whale */}
          <div>
            <div style={{ fontSize: '12px', color: 'rgba(255, 255, 255, 0.6)', marginBottom: '12px', fontWeight: '600' }}>
              🐋 BALINALAR (&gt;$1M)
            </div>
            <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
              <div style={{ flex: 1, padding: '10px', background: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '6px' }}>
                <div style={{ fontSize: '10px', color: '#10B981', marginBottom: '2px' }}>Alış</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#10B981' }}>{cvdData.sizeAnalysis.whale.buyVolume.toFixed(0)}</div>
              </div>
              <div style={{ flex: 1, padding: '10px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '6px' }}>
                <div style={{ fontSize: '10px', color: '#EF4444', marginBottom: '2px' }}>Satış</div>
                <div style={{ fontSize: '14px', fontWeight: '700', color: '#EF4444' }}>{cvdData.sizeAnalysis.whale.sellVolume.toFixed(0)}</div>
              </div>
            </div>
            <div style={{ fontSize: '11px', color: 'rgba(255, 255, 255, 0.5)' }}>
              Oran: {cvdData.sizeAnalysis.whale.percentage.toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Educational Info */}
      <div style={{
        padding: '24px',
        background: 'rgba(139, 92, 246, 0.1)',
        border: '1px solid rgba(139, 92, 246, 0.2)',
        borderRadius: '12px',
        fontSize: '13px',
        color: 'rgba(255, 255, 255, 0.7)',
        lineHeight: '1.8'
      }}>
        <div style={{ fontWeight: '700', color: '#8B5CF6', marginBottom: '16px', fontSize: '16px' }}>
          💡 CVD (Cumulative Volume Delta) - Profesyonellerin Gizli Silahı
        </div>

        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontWeight: '600', color: '#FFFFFF', marginBottom: '8px' }}>
            🎯 CVD Nedir?
          </div>
          <div>
            CVD (Cumulative Volume Delta), her işlemin alış mı satış mı olduğunu tespit edip, bunları birikimli olarak toplayarak
            piyasadaki GERÇEK alış/satış baskısını gösteren profesyonel bir göstergedir. Normal fiyat grafikleri sadece fiyat hareketini
            gösterir, ama CVD FİYATIN ARKASINDA ne olduğunu gösterir.
          </div>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontWeight: '600', color: '#FFFFFF', marginBottom: '8px' }}>
            🔥 Neden Bu Kadar Kritik?
          </div>
          <div>
            • <strong style={{ color: '#10B981' }}>Pozitif CVD + Yükselen Fiyat</strong> = Sağlıklı yükseliş, kurumsal para giriyor ✅<br/>
            • <strong style={{ color: '#EF4444' }}>Negatif CVD + Yükselen Fiyat</strong> = SAHTE YUKSELIŞ! Kurumsal satıyor, düşüş yakın ⚠️<br/>
            • <strong style={{ color: '#10B981' }}>Pozitif CVD + Düşen Fiyat</strong> = DİP ALIM FIRSATI! Kurumsal topluyor, yükseliş gelecek 🎯<br/>
            • <strong style={{ color: '#EF4444' }}>Negatif CVD + Düşen Fiyat</strong> = Güçlü düşüş, bekle 🛑
          </div>
        </div>

        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontWeight: '600', color: '#FFFFFF', marginBottom: '8px' }}>
            📊 Büyüklük Analizi (Kim Alıyor/Satıyor?)
          </div>
          <div>
            • <strong style={{ color: '#6B7280' }}>🐟 RETAIL (&lt;$100k)</strong>: Küçük yatırımcılar - Genelde duygusal işlem yapar<br/>
            • <strong style={{ color: '#3B82F6' }}>🏛️ KURUMSAL ($100k-$1M)</strong>: Profesyonel trader'lar ve fonlar - Stratejik hareket eder<br/>
            • <strong style={{ color: '#EAB308' }}>🐋 BALINALAR (&gt;$1M)</strong>: Büyük sermaye - Piyasayı yönlendirir, takip et!
          </div>
        </div>

        <div style={{
          background: 'rgba(234, 179, 8, 0.15)',
          padding: '12px',
          borderRadius: '8px',
          border: '1px solid rgba(234, 179, 8, 0.3)'
        }}>
          <div style={{ fontWeight: '600', color: '#EAB308', marginBottom: '6px' }}>
            💰 CZ (Binance Sahibi) Neden CVD Kullanıyor?
          </div>
          <div style={{ fontSize: '12px' }}>
            Çünkü CVD, piyasanın GERÇEKTEn nereye gittiğini gösterir. Fiyat manipüle edilebilir ama hacim manipüle edilemez!
            BTC $97k'ya çıktığında CVD düşüyordu = CZ satış yaptı, sonra düzeltme geldi. Normal trader'lar bunu GÖREMEDİ,
            CZ gördü. Bu yüzden CVD, kurumsal seviye bir araçtır ve profesyonel trader'ların vazgeçilmez göstergesidir.
          </div>
        </div>
      </div>
    </div>
  );
}
