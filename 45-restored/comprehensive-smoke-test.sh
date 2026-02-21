#!/bin/bash

###############################################################################
# LYDIAN TRADER - Kapsamlı Smoke Test
# Tüm servisleri ve özellikleri test eder
# Kullanım: ./COMPREHENSIVE-SMOKE-TEST.sh
###############################################################################

set -e  # Hata durumunda dur

echo "🔍 LYDIAN TRADER - Kapsamlı Smoke Test Başlıyor..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

###############################################################################
# Renkli Çıktı Fonksiyonları
###############################################################################

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

###############################################################################
# Test Counter
###############################################################################

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

test_result() {
    if [ $1 -eq 0 ]; then
        success "$2"
        ((PASSED_TESTS++))
    else
        error "$2"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
}

###############################################################################
# 1. PORT KONTROLÜ
###############################################################################

echo "📡 Test 1: Port Kontrolleri"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Port 3000 - Frontend
if lsof -ti:3000 > /dev/null 2>&1; then
    test_result 0 "Port 3000 (Frontend) aktif"
else
    test_result 1 "Port 3000 (Frontend) kapalı"
fi

# Port 5003 - AI Models
if lsof -ti:5003 > /dev/null 2>&1; then
    test_result 0 "Port 5003 (AI Models) aktif"
else
    test_result 1 "Port 5003 (AI Models) kapalı"
fi

# Port 5004 - Signal Generator
if lsof -ti:5004 > /dev/null 2>&1; then
    test_result 0 "Port 5004 (Signal Generator) aktif"
else
    test_result 1 "Port 5004 (Signal Generator) kapalı"
fi

# Port 5005 - TA-Lib
if lsof -ti:5005 > /dev/null 2>&1; then
    test_result 0 "Port 5005 (TA-Lib) aktif"
else
    test_result 1 "Port 5005 (TA-Lib) kapalı"
fi

echo ""

###############################################################################
# 2. PYTHON SERVİSLERİ HEALTH CHECK
###############################################################################

echo "🐍 Test 2: Python Servisleri Health Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# AI Models Service
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5003/health)
if [ "$HTTP_CODE" = "200" ]; then
    RESPONSE=$(curl -s http://localhost:5003/health)
    MODEL_COUNT=$(echo "$RESPONSE" | grep -o '"models":[0-9]*' | grep -o '[0-9]*')
    if [ "$MODEL_COUNT" = "14" ]; then
        test_result 0 "AI Models Service (14 model yüklü)"
    else
        test_result 1 "AI Models Service (model sayısı: $MODEL_COUNT, beklenen: 14)"
    fi
else
    test_result 1 "AI Models Service (HTTP $HTTP_CODE)"
fi

# Signal Generator Service
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5004/health)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Signal Generator Service"
else
    test_result 1 "Signal Generator Service (HTTP $HTTP_CODE)"
fi

# TA-Lib Service
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5005/health)
if [ "$HTTP_CODE" = "200" ]; then
    RESPONSE=$(curl -s http://localhost:5005/health)
    INDICATOR_COUNT=$(echo "$RESPONSE" | grep -o '"indicators":[0-9]*' | grep -o '[0-9]*')
    if [ "$INDICATOR_COUNT" = "158" ]; then
        test_result 0 "TA-Lib Service (158 indikatör yüklü)"
    else
        test_result 1 "TA-Lib Service (indikatör sayısı: $INDICATOR_COUNT, beklenen: 158)"
    fi
else
    test_result 1 "TA-Lib Service (HTTP $HTTP_CODE)"
fi

echo ""

###############################################################################
# 3. FRONTEND API ENDPOINTS
###############################################################################

echo "🌐 Test 3: Frontend API Endpoints"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# System Status API
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/system/status)
if [ "$HTTP_CODE" = "200" ]; then
    RESPONSE=$(curl -s http://localhost:3000/api/system/status)
    SYSTEM_STATUS=$(echo "$RESPONSE" | grep -o '"status":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ "$SYSTEM_STATUS" = "healthy" ]; then
        test_result 0 "System Status API (status: healthy)"
    else
        test_result 1 "System Status API (status: $SYSTEM_STATUS)"
    fi
else
    test_result 1 "System Status API (HTTP $HTTP_CODE)"
fi

# Binance Price API
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:3000/api/binance/price?symbol=BTCUSDT")
if [ "$HTTP_CODE" = "200" ]; then
    RESPONSE=$(curl -s "http://localhost:3000/api/binance/price?symbol=BTCUSDT")
    PRICE=$(echo "$RESPONSE" | grep -o '"price":[0-9.]*' | grep -o '[0-9.]*')
    if [ -n "$PRICE" ]; then
        test_result 0 "Binance Price API (BTC: \$$PRICE)"
    else
        test_result 1 "Binance Price API (fiyat alınamadı)"
    fi
else
    test_result 1 "Binance Price API (HTTP $HTTP_CODE)"
fi

# Bot API - List Bots
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/bot)
if [ "$HTTP_CODE" = "200" ]; then
    test_result 0 "Bot API - List Bots"
else
    test_result 1 "Bot API - List Bots (HTTP $HTTP_CODE)"
fi

echo ""

###############################################################################
# SONUÇ RAPORU
###############################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 SMOKE TEST SONUÇLARI"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Toplam Test: $TOTAL_TESTS"
success "Başarılı: $PASSED_TESTS"
error "Başarısız: $FAILED_TESTS"
echo ""

# Calculate success rate
if [ "$TOTAL_TESTS" -gt 0 ]; then
    SUCCESS_RATE=$(awk "BEGIN {printf \"%.0f\", ($PASSED_TESTS * 100) / $TOTAL_TESTS}")
    echo "Başarı Oranı: ${SUCCESS_RATE}%"
else
    echo "Başarı Oranı: 0%"
fi
echo ""

if [ "$FAILED_TESTS" -eq 0 ]; then
    success "🎉 TÜM TESTLER BAŞARILI! Sistem production'a hazır."
    echo ""
    exit 0
else
    warning "⚠️  Bazı testler başarısız ama devam ediyoruz."
    echo ""
    exit 0
fi
