#!/bin/bash

# Development Smoke Test - Tüm sayfaları tara ve mock veri kontrolü yap

echo "🔍 LYDIAN TRADER - Development Smoke Test"
echo "=========================================="
echo ""

BASE_URL="http://localhost:3000"

# Sayfalar listesi
pages=(
  "/"
  "/dashboard"
  "/crypto"
  "/stocks"
  "/portfolio"
  "/watchlist"
  "/market-analysis"
  "/live-trading"
  "/quantum-pro"
  "/futures-bot"
  "/bot-management"
  "/ai-testing"
  "/ai-chat"
  "/signals"
  "/backtesting"
  "/risk-management"
  "/auto-trading"
  "/ai-control-center"
)

echo "📋 Testing ${#pages[@]} pages for availability..."
echo ""

failed=0
passed=0

for page in "${pages[@]}"; do
  echo -n "Testing $page ... "

  response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL$page" --max-time 5 2>/dev/null)

  if [ "$response" = "200" ]; then
    echo "✅ OK (HTTP $response)"
    ((passed++))
  else
    echo "❌ FAILED (HTTP $response)"
    ((failed++))
  fi
done

echo ""
echo "=========================================="
echo "📊 Test Summary:"
echo "   ✅ Passed: $passed"
echo "   ❌ Failed: $failed"
echo "   📈 Success Rate: $(( passed * 100 / ${#pages[@]} ))%"
echo ""

if [ $failed -eq 0 ]; then
  echo "🎉 All pages are accessible!"
  exit 0
else
  echo "⚠️  Some pages failed accessibility test"
  exit 1
fi
