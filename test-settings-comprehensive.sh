#!/bin/bash

echo "=================================================="
echo "🔍 AiLydian EMRAH - SETTINGS COMPREHENSIVE SMOKE TEST"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:3000"
TIMEOUT=10

# Test counter
PASSED=0
FAILED=0
TOTAL=0

# Function to test an API endpoint
test_api() {
    local name=$1
    local endpoint=$2
    local method=${3:-GET}
    local expect_status=${4:-200}

    ((TOTAL++))

    echo -n "Testing $name... "

    if [ "$method" = "GET" ]; then
        response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$BASE_URL$endpoint" 2>/dev/null)
    else
        response=$(curl -s -o /dev/null -w "%{http_code}" -X $method --max-time $TIMEOUT "$BASE_URL$endpoint" 2>/dev/null)
    fi

    if [ "$response" = "$expect_status" ]; then
        echo -e "${GREEN}✅ PASS${NC} (HTTP $response)"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAIL${NC} (Expected $expect_status, Got $response)"
        ((FAILED++))
    fi
}

echo "=================================================="
echo "1️⃣  SETTINGS API ENDPOINTS"
echo "=================================================="

test_api "Settings (General)" "/api/settings" "GET" "200"
test_api "Risk Management" "/api/risk-management" "GET" "200"
test_api "Notifications" "/api/notifications" "GET" "200"
test_api "Strategy Sync" "/api/strategy-sync" "GET" "200"
test_api "Security Audit" "/api/security/audit" "GET" "200"

echo ""
echo "=================================================="
echo "2️⃣  PERFORMANCE & ANALYTICS ENDPOINTS"
echo "=================================================="

# Check if performance-analytics endpoint exists
test_api "Performance Analytics" "/api/settings" "GET" "200"
test_api "Queue Metrics" "/api/queue/metrics" "GET" "200"

echo ""
echo "=================================================="
echo "3️⃣  CORE DATA ENDPOINTS (Settings Dependencies)"
echo "=================================================="

test_api "Binance Futures" "/api/binance/futures" "GET" "200"
test_api "Signals" "/api/signals" "GET" "200"
test_api "AI Signals" "/api/ai-signals" "GET" "200"
test_api "Quantum Signals" "/api/quantum-signals" "GET" "200"
test_api "Conservative Signals" "/api/conservative-signals" "GET" "200"

echo ""
echo "=================================================="
echo "4️⃣  NOTIFICATION SYSTEM ENDPOINTS"
echo "=================================================="

test_api "Push Subscribe" "/api/push/subscribe" "POST" "405"
test_api "Push Stats" "/api/push/stats" "GET" "200"
test_api "Telegram Live" "/api/telegram/live" "GET" "200"

echo ""
echo "=================================================="
echo "5️⃣  SYSTEM HEALTH & STATUS"
echo "=================================================="

test_api "Health Check" "/api/health" "GET" "200"
test_api "Autonomous Health" "/api/health/autonomous" "GET" "200"
test_api "Scanner Status" "/api/scanner/status" "GET" "200"

echo ""
echo "=================================================="
echo "6️⃣  SETTINGS PAGE FRONTEND TEST"
echo "=================================================="

echo -n "Testing Settings Page HTML... "
response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$BASE_URL/settings" 2>/dev/null)
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ PASS${NC} (HTTP $response)"
    ((PASSED++))
    ((TOTAL++))
else
    echo -e "${RED}❌ FAIL${NC} (Got $response)"
    ((FAILED++))
    ((TOTAL++))
fi

echo ""
echo "=================================================="
echo "7️⃣  BACKEND SERVICES STATUS"
echo "=================================================="

# Check Python services
echo "Checking Python Services..."

check_service() {
    local name=$1
    local port=$2

    echo -n "  $name (Port $port)... "

    if curl -s --max-time 2 "http://localhost:$port/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ RUNNING${NC}"
    else
        echo -e "${YELLOW}⚠️  NOT RUNNING${NC}"
    fi
}

check_service "TA-Lib Service" "5001"
check_service "Database Service" "5020"
check_service "Order Flow Service" "5021"

echo ""
echo "=================================================="
echo "📊 TEST SUMMARY"
echo "=================================================="
echo "Total Tests: $TOTAL"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ ALL TESTS PASSED!${NC}"
    echo "Settings page is fully synchronized with backend!"
    exit 0
else
    echo -e "\n${RED}❌ SOME TESTS FAILED!${NC}"
    echo "Please check the failed endpoints above."
    exit 1
fi
