#!/bin/bash

# SARDAG TRADING SCANNER - COMPREHENSIVE SMOKE TEST
# Tests all endpoints and verifies REAL DATA (no mocks)

echo "🔥 SARDAG Trading Scanner - Comprehensive Smoke Test"
echo "======================================================"
echo ""

BASE_URL="http://localhost:3000"
PASSED=0
FAILED=0
WARNINGS=0

# Function to test endpoint
test_endpoint() {
  local name="$1"
  local endpoint="$2"
  local timeout="${3:-10}"
  local expected_field="${4:-success}"

  echo -n "🔍 Testing: $name ... "

  response=$(timeout $timeout curl -s "$BASE_URL$endpoint" 2>&1)
  exit_code=$?

  if [ $exit_code -eq 124 ]; then
    echo "❌ TIMEOUT (>${timeout}s)"
    ((FAILED++))
    return 1
  elif [ $exit_code -ne 0 ]; then
    echo "❌ FAILED (connection error)"
    ((FAILED++))
    return 1
  fi

  # Check if response contains expected field
  if echo "$response" | grep -q "\"$expected_field\""; then
    # Check for mock/demo indicators
    if echo "$response" | grep -qi "mock\|demo\|fake\|placeholder\|approximate"; then
      echo "⚠️  WARNING - Contains mock/fallback data"
      ((WARNINGS++))
      echo "   Response snippet: $(echo "$response" | head -c 200)..."
      return 2
    else
      echo "✅ PASSED (real data)"
      ((PASSED++))
      return 0
    fi
  else
    echo "❌ FAILED (invalid response)"
    echo "   Response snippet: $(echo "$response" | head -c 200)..."
    ((FAILED++))
    return 1
  fi
}

echo "📊 CORE DATA ENDPOINTS:"
echo "----------------------"
test_endpoint "Binance Futures" "/api/binance/futures" 15 "success"
test_endpoint "Signals" "/api/signals?limit=50" 10 "success"
test_endpoint "AI Signals" "/api/ai-signals" 15 "success"
test_endpoint "Quantum Signals" "/api/quantum-signals" 15 "success"
test_endpoint "Conservative Signals" "/api/conservative-signals" 10 "success"
test_endpoint "Market Correlation" "/api/market-correlation?limit=50" 15 "success"
test_endpoint "Breakout-Retest" "/api/breakout-retest" 10 "success"
test_endpoint "Unified Signals" "/api/unified-signals?limit=50" 15 "success"
test_endpoint "Omnipotent Futures" "/api/omnipotent-futures?limit=50" 15 "success"
test_endpoint "BTC-ETH Analysis" "/api/btc-eth-analysis" 10 "success"

echo ""
echo "📈 TRADITIONAL MARKETS:"
echo "----------------------"
test_endpoint "Traditional Markets" "/api/traditional-markets" 15 "success"
test_endpoint "Crypto News" "/api/crypto-news" 10 "success"

echo ""
echo "🔧 TECHNICAL ANALYSIS:"
echo "---------------------"
test_endpoint "Ta-Lib Analysis (BTC)" "/api/talib-analysis/BTCUSDT" 10 "success"

echo ""
echo "📡 SYSTEM HEALTH:"
echo "----------------"
test_endpoint "News Risk Alerts" "/api/news-risk-alerts" 10 "success"

echo ""
echo "======================================================"
echo "📊 SMOKE TEST RESULTS:"
echo "======================================================"
echo "✅ Passed:   $PASSED"
echo "❌ Failed:   $FAILED"
echo "⚠️  Warnings: $WARNINGS (mock/fallback data detected)"
echo ""

TOTAL=$((PASSED + FAILED))
if [ $TOTAL -gt 0 ]; then
  SUCCESS_RATE=$((PASSED * 100 / TOTAL))
  echo "Success Rate: ${SUCCESS_RATE}%"
  echo ""

  if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
      echo "🎉 ALL TESTS PASSED - 100% REAL DATA!"
      exit 0
    else
      echo "⚠️  TESTS PASSED BUT WITH MOCK DATA WARNINGS"
      echo "   Some endpoints are using fallback/mock data."
      echo "   Check .env.local for missing API keys."
      exit 2
    fi
  else
    echo "❌ SOME TESTS FAILED"
    exit 1
  fi
else
  echo "❌ NO TESTS RUN"
  exit 1
fi
