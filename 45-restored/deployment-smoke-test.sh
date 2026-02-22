#!/bin/bash

# Deployment Smoke Test - Vercel & Railway
# Ensures production deployment passes all critical checks

set -e

echo "🚀 Deployment Smoke Test - Starting..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0

# Test function
test_endpoint() {
  local name=$1
  local url=$2
  local expected_code=${3:-200}

  echo -n "Testing $name... "

  response_code=$(curl -s -o /dev/null -w "%{http_code}" "$url" --max-time 10 || echo "000")

  if [ "$response_code" = "$expected_code" ]; then
    echo -e "${GREEN}✅ PASS${NC} (HTTP $response_code)"
    ((PASS_COUNT++))
    return 0
  else
    echo -e "${RED}❌ FAIL${NC} (HTTP $response_code, expected $expected_code)"
    ((FAIL_COUNT++))
    return 1
  fi
}

# Get Vercel deployment URL from GitHub API
echo "📋 Step 1: Checking GitHub deployment status..."
GITHUB_REPO="lydiansoftware/borsa"
LATEST_COMMIT=$(git rev-parse HEAD)

echo "   Latest commit: $LATEST_COMMIT"
echo ""

# Check for vercel.json or .vercel directory for deployment URL
if [ -d ".vercel" ]; then
  VERCEL_URL=$(cat .vercel/project.json 2>/dev/null | grep -o '"projectId":"[^"]*"' | cut -d'"' -f4 || echo "")
  if [ -n "$VERCEL_URL" ]; then
    echo "✅ Vercel project ID found: $VERCEL_URL"
  fi
fi

# Try to get production URL from common patterns
PRODUCTION_URLS=(
  "https://borsa-lydiansoftware.vercel.app"
  "https://borsa.vercel.app"
  "https://lydian-trader.vercel.app"
  "https://borsa-git-main-lydiansoftware.vercel.app"
)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Step 2: Testing production URLs..."
echo ""

WORKING_URL=""
for url in "${PRODUCTION_URLS[@]}"; do
  echo -n "   Trying $url... "
  if curl -s -o /dev/null -w "%{http_code}" "$url" --max-time 5 | grep -q "200"; then
    echo -e "${GREEN}✅ ACCESSIBLE${NC}"
    WORKING_URL="$url"
    break
  else
    echo -e "${YELLOW}⏭️  SKIP${NC}"
  fi
done

if [ -z "$WORKING_URL" ]; then
  echo ""
  echo -e "${RED}❌ No working production URL found${NC}"
  echo "   Please check:"
  echo "   1. Vercel dashboard: https://vercel.com/dashboard"
  echo "   2. GitHub deployment status"
  echo ""
  echo "   The deployment may still be building..."
  exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔍 Step 3: Running critical path smoke tests..."
echo ""
echo "   Using production URL: $WORKING_URL"
echo ""

# Critical path tests
test_endpoint "Homepage" "$WORKING_URL/" 200
test_endpoint "Login page" "$WORKING_URL/login" 200
test_endpoint "Dashboard" "$WORKING_URL/dashboard" 200
test_endpoint "AI Testing Center" "$WORKING_URL/ai-testing" 200
test_endpoint "Bot Test Page" "$WORKING_URL/bot-test" 200
test_endpoint "Trading Signals" "$WORKING_URL/signals" 200
test_endpoint "AI Control Center" "$WORKING_URL/ai-control-center" 200

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔌 Step 4: Testing API endpoints..."
echo ""

test_endpoint "Market API - Top 100" "$WORKING_URL/api/market/top100" 200
test_endpoint "AI Predict API" "$WORKING_URL/api/ai/predict" 405 # POST only
test_endpoint "Trading Signals API" "$WORKING_URL/api/trading/signals" 405 # POST only
test_endpoint "AI Models API" "$WORKING_URL/api/ai/models" 200

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Step 5: Verifying deployment metadata..."
echo ""

# Check build metadata
echo -n "   Build info... "
if curl -s "$WORKING_URL" | grep -q "next-head"; then
  echo -e "${GREEN}✅ Next.js detected${NC}"
  ((PASS_COUNT++))
else
  echo -e "${YELLOW}⚠️  WARNING${NC} - Next.js not detected"
fi

echo -n "   Turbopack... "
if grep -q "turbo" package.json 2>/dev/null; then
  echo -e "${GREEN}✅ Configured${NC}"
  ((PASS_COUNT++))
else
  echo -e "${YELLOW}ℹ️  Not configured${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🏁 Final Results:"
echo ""
echo "   Total Passed: ${GREEN}$PASS_COUNT${NC}"
echo "   Total Failed: ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${GREEN}✅ ALL TESTS PASSED - DEPLOYMENT READY FOR PRODUCTION${NC}"
  echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo "🚀 Production URL: $WORKING_URL"
  echo ""
  echo "   Next steps:"
  echo "   1. Visit Vercel dashboard to promote to production domain"
  echo "   2. Configure custom domain if needed"
  echo "   3. Set environment variables in Vercel dashboard"
  echo ""
  exit 0
else
  echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${RED}❌ DEPLOYMENT HAS ISSUES - PLEASE REVIEW${NC}"
  echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo ""
  echo "   Check Vercel deployment logs:"
  echo "   https://vercel.com/dashboard"
  echo ""
  exit 1
fi
