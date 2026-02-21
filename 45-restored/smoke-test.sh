#!/bin/bash

echo "🚀 LyDian Trader - Complete System Smoke Test"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to test endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}

    echo -n "Testing $name... "

    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)

    if [ "$response" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASSED${NC} (HTTP $response)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $response, expected $expected_status)"
        ((FAILED++))
        return 1
    fi
}

# Check if server is running
echo "📡 Checking server status..."
if ! lsof -ti:3000 > /dev/null 2>&1; then
    echo -e "${RED}✗ Server not running on port 3000${NC}"
    echo "Starting server..."
    cd ~/Desktop/borsa
    npm run dev > /dev/null 2>&1 &
    sleep 5
fi

echo ""
echo "🌐 Testing Frontend Pages..."
echo "-----------------------------"
test_endpoint "Home Page" "http://localhost:3000"
test_endpoint "Login Page" "http://localhost:3000/login"
test_endpoint "Dashboard" "http://localhost:3000/dashboard"
test_endpoint "Crypto Page" "http://localhost:3000/crypto"
test_endpoint "Stocks Page" "http://localhost:3000/stocks"
test_endpoint "Portfolio Page" "http://localhost:3000/portfolio"
test_endpoint "Watchlist Page" "http://localhost:3000/watchlist"

echo ""
echo "🔌 Testing Backend APIs..."
echo "--------------------------"
test_endpoint "Market Data API" "http://localhost:3000/api/market/crypto"
test_endpoint "Location API" "http://localhost:3000/api/location"
test_endpoint "Quantum Pro Signals API" "http://localhost:3000/api/quantum-pro/signals"

echo ""
echo "🧪 Testing AI Components..."
echo "---------------------------"
# Check if service files exist
if [ -f "src/services/ai/QuantumProEngine.ts" ]; then
    echo -e "${GREEN}✓${NC} QuantumProEngine.ts exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} QuantumProEngine.ts missing"
    ((FAILED++))
fi

if [ -f "src/services/ai/BacktestingEngine.ts" ]; then
    echo -e "${GREEN}✓${NC} BacktestingEngine.ts exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} BacktestingEngine.ts missing"
    ((FAILED++))
fi

if [ -f "src/services/ai/RiskManagementModule.ts" ]; then
    echo -e "${GREEN}✓${NC} RiskManagementModule.ts exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} RiskManagementModule.ts missing"
    ((FAILED++))
fi

echo ""
echo "📦 Testing Components..."
echo "------------------------"
if [ -f "src/components/Navigation.tsx" ]; then
    echo -e "${GREEN}✓${NC} Navigation component exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Navigation component missing"
    ((FAILED++))
fi

if [ -f "src/components/ui/Logo.tsx" ]; then
    echo -e "${GREEN}✓${NC} Logo component exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Logo component missing"
    ((FAILED++))
fi

if [ -f "src/components/LoginMap.tsx" ]; then
    echo -e "${GREEN}✓${NC} LoginMap component exists"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} LoginMap component missing"
    ((FAILED++))
fi

echo ""
echo "=============================================="
echo "📊 Test Results"
echo "=============================================="
echo -e "Total Tests: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo "System is fully operational and ready for use."
    exit 0
else
    echo -e "${YELLOW}⚠️  Some tests failed. Check the output above.${NC}"
    exit 1
fi