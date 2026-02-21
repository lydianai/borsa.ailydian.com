#!/bin/bash

# ====================================
# RAPIDAPI TEST SCRIPT
# ====================================
# Bu script RapidAPI key'lerinizi test eder
# Kullanım: chmod +x test-rapidapi.sh && ./test-rapidapi.sh

# Load .env.local
if [ -f .env.local ]; then
    export $(grep -v '^#' .env.local | xargs)
fi

echo "🧪 RapidAPI Entegrasyon Testi Başlıyor..."
echo "=========================================="
echo ""

# Check if RAPIDAPI_KEY is set
if [ "$RAPIDAPI_KEY" = "your_rapidapi_key_here" ] || [ -z "$RAPIDAPI_KEY" ]; then
    echo "❌ HATA: RAPIDAPI_KEY bulunamadı!"
    echo "📝 Lütfen .env.local dosyasında RAPIDAPI_KEY=xxx satırını doldurun"
    echo ""
    echo "Nereden alınır:"
    echo "1. https://rapidapi.com adresine gidin"
    echo "2. Login olun"
    echo "3. https://rapidapi.com/developer/security adresinden API key alın"
    exit 1
fi

echo "✅ API Key bulundu: ${RAPIDAPI_KEY:0:10}..."
echo ""

# Test 1: Harem Altın API
echo "📊 Test 1: Harem Altın API"
echo "----------------------------"
echo "Endpoint: https://$RAPIDAPI_HAREM_HOST/"

GOLD_RESPONSE=$(curl -s \
  -H "x-rapidapi-key: $RAPIDAPI_KEY" \
  -H "x-rapidapi-host: $RAPIDAPI_HAREM_HOST" \
  "https://$RAPIDAPI_HAREM_HOST/" \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$GOLD_RESPONSE" | grep "HTTP_CODE" | cut -d':' -f2)
RESPONSE_BODY=$(echo "$GOLD_RESPONSE" | sed '/HTTP_CODE/d')

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Başarılı! (HTTP 200)"
    echo "📦 Response:"
    echo "$RESPONSE_BODY" | head -20
else
    echo "❌ Hata! (HTTP $HTTP_CODE)"
    echo "📦 Response:"
    echo "$RESPONSE_BODY"
fi
echo ""
echo ""

# Test 2: Crypto News API
echo "📰 Test 2: Crypto News API"
echo "----------------------------"
echo "Endpoint: https://$RAPIDAPI_NEWS_HOST/news/top/10"

NEWS_RESPONSE=$(curl -s \
  -H "x-rapidapi-key: $RAPIDAPI_KEY" \
  -H "x-rapidapi-host: $RAPIDAPI_NEWS_HOST" \
  "https://$RAPIDAPI_NEWS_HOST/news/top/10" \
  -w "\nHTTP_CODE:%{http_code}")

HTTP_CODE=$(echo "$NEWS_RESPONSE" | grep "HTTP_CODE" | cut -d':' -f2)
RESPONSE_BODY=$(echo "$NEWS_RESPONSE" | sed '/HTTP_CODE/d')

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Başarılı! (HTTP 200)"
    echo "📦 Response:"
    echo "$RESPONSE_BODY" | head -30
else
    echo "❌ Hata! (HTTP $HTTP_CODE)"
    echo "📦 Response:"
    echo "$RESPONSE_BODY"
fi
echo ""
echo ""

# Summary
echo "=========================================="
echo "✅ Test tamamlandı!"
echo ""
echo "📌 Sonraki Adımlar:"
echo "1. Her iki API de 200 döndü mü? → Devam edelim!"
echo "2. Hata var mı? → API key'i kontrol edin"
echo "3. 403 Forbidden → API'ye subscribe oldunuz mu?"
echo "4. 429 Too Many Requests → Rate limit aşıldı, bekleyin"
echo ""
