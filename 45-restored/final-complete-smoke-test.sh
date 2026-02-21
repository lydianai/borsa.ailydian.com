#!/bin/bash

echo "🚀 LYDIAN TRADER - FINAL COMPLETE SMOKE TEST"
echo "=============================================="
echo ""

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

test_endpoint() {
    local name=$1
    local url=$2
    echo -n "Testing $name... "
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ (HTTP $response)${NC}"
        ((FAILED++))
    fi
}

echo "🌐 ALL PAGES..."
echo "---------------"
test_endpoint "Home" "http://localhost:3000"
test_endpoint "Login" "http://localhost:3000/login"
test_endpoint "Dashboard" "http://localhost:3000/dashboard"
test_endpoint "AI Chat Assistant" "http://localhost:3000/ai-chat"
test_endpoint "Crypto" "http://localhost:3000/crypto"
test_endpoint "Stocks" "http://localhost:3000/stocks"
test_endpoint "Portfolio" "http://localhost:3000/portfolio"
test_endpoint "Watchlist" "http://localhost:3000/watchlist"
test_endpoint "Bot Management" "http://localhost:3000/bot-management"
test_endpoint "Signals" "http://localhost:3000/signals"
test_endpoint "Quantum Pro" "http://localhost:3000/quantum-pro"
test_endpoint "Backtesting" "http://localhost:3000/backtesting"
test_endpoint "Risk Management" "http://localhost:3000/risk-management"
test_endpoint "Market Analysis" "http://localhost:3000/market-analysis"
test_endpoint "Settings" "http://localhost:3000/settings"

echo ""
echo "🔌 ALL APIs..."
echo "--------------"
test_endpoint "Market Data" "http://localhost:3000/api/market/crypto"
test_endpoint "Quantum Signals" "http://localhost:3000/api/quantum-pro/signals"
test_endpoint "Quantum Monitor" "http://localhost:3000/api/quantum-pro/monitor"
test_endpoint "Quantum Bots" "http://localhost:3000/api/quantum-pro/bots"
test_endpoint "Location" "http://localhost:3000/api/location"

echo ""
echo "=============================================="
echo "📊 FINAL RESULTS"
echo "=============================================="
echo -e "Total: $((PASSED + FAILED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL $PASSED TESTS PASSED!${NC}"
    echo "✅ System fully operational with AI Assistant!"
    exit 0
else
    echo -e "${YELLOW}⚠️  $FAILED test(s) failed${NC}"
    exit 1
fi
