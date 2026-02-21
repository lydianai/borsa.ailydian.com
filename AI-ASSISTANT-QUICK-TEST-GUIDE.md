# AI ASSISTANT - QUICK TEST GUIDE

**Test Edilecek Feature:** AI Assistant'ın 600+ coin tanıma ve 6 strateji entegrasyonu

---

## TEST 1: TRB COIN SORGUSU

### Input:
```
TRB alınır mı?
```

### Beklenen Output İçeriği:
- ✅ TRB fiyat bilgisi gösterilmeli
- ✅ 24h değişim ve hacim gösterilmeli
- ✅ AL/SAT/BEKLE kararı verilmeli
- ✅ 6 stratejiden gelen sinyaller listelenmeli
- ✅ "Bilgim yok" OLMAMALI

### API Flow:
1. `/api/binance/futures` → TRB datasını çeker
2. `/api/signals` → Manuel sinyal
3. `/api/ai-signals` → AI sinyal
4. `/api/conservative-signals` → Conservative sinyal
5. `/api/quantum-signals` → Quantum sinyal
6. `/api/market-correlation` → Correlation sinyal
7. `/api/breakout-retest` → Breakout sinyal
8. Groq AI → Final analiz

---

## TEST 2: LEVER COIN SORGUSU

### Input:
```
LEVER alsam mı?
```

### Beklenen Output İçeriği:
- ✅ LEVER fiyat bilgisi
- ✅ Strateji sonuçları
- ✅ Detaylı açıklama
- ✅ Risk uyarıları

---

## TEST 3: EXOTIC COIN SORGUSU

### Input:
```
RUNE hakkında ne düşünüyorsun?
```

### Beklenen:
- ✅ RUNE datasını bulmalı
- ✅ 6 stratejiyi analiz etmeli
- ✅ "600+ coin içinden RUNE'u analiz ettim" gibi mesaj

---

## TEST 4: GENEL PİYASA SORGUSU

### Input:
```
Piyasa nasıl?
```

### Beklenen Output İçeriği:
- ✅ "612 coin takip ediliyor" (veya güncel sayı)
- ✅ "TÜM Binance Futures USDT çiftleri"
- ✅ Top gainers listesi
- ✅ En yüksek hacimli coinler

---

## TEST 5: TÜRKÇE SORU ÇEŞİTLİLİĞİ

### Inputs:
```
1. "TRB almalı mıyım?"
2. "LEVER satılır mı?"
3. "RUNE beklemeli miyim?"
4. "APE hakkında bilgi ver"
```

### Beklenen:
- ✅ Tüm varyasyonlarda coin tanınmalı
- ✅ extractSymbol() fonksiyonu çalışmalı

---

## HATA SENARYOLARI

### Test 6: Olmayan Coin
**Input:** "FAKECOIN alınır mı?"

**Beklenen:**
```
FAKECOIN hakkında detaylı bilgi bulunamadı.
Bu coin Binance Futures'da olmayabilir veya çok düşük hacimli olabilir.
```

---

## CONSOLE LOG KONTROLLERİ

Test sırasında browser console'da şunlar görülmeli:

```
[AI Assistant] Fetching all strategies for TRBUSDT...
[AI Assistant] Manual signals unavailable (veya available)
[AI Assistant] AI signals available
[AI Assistant] Conservative signals available
[AI Assistant] Quantum signals available
[AI Assistant] Correlation signals available
[AI Assistant] Breakout signals available
```

---

## BAŞARILI TEST KRİTERLERİ

- [ ] TRB tanınıyor ve analiz ediliyor
- [ ] LEVER tanınıyor ve analiz ediliyor
- [ ] 600+ coin'den herhangi biri sorulduğunda cevap veriyor
- [ ] 6 strateji sonucu gösteriliyor
- [ ] "Bilgim yok" hatası YOK
- [ ] Response time < 5 saniye
- [ ] Türkçe varyasyonlar çalışıyor
- [ ] Groq AI detaylı açıklama yapıyor

---

## HIZLI TEST KOMUTU

```bash
# API endpoint'i test et
curl -X POST http://localhost:3000/api/ai-assistant \
  -H "Content-Type: application/json" \
  -d '{"message": "TRB alınır mı?", "history": []}'
```

---

## PRODUCTION TEST URL

```
https://sardag-emrah.vercel.app/api/ai-assistant
```

**POST Request Body:**
```json
{
  "message": "TRB alınır mı?",
  "history": []
}
```

---

**Not:** Tüm testler PASSED olmalı. Herhangi bir test FAIL olursa rapor et.
