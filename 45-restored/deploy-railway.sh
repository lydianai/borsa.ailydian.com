#!/bin/bash

# Railway Deployment Script for Python AI Services
# Author: Claude Code
# Date: 2025-10-02

echo "🛤️  RAILWAY DEPLOYMENT - Python AI Services"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo -e "${RED}❌ Railway CLI not found${NC}"
    echo "Installing Railway CLI..."
    npm install -g @railway/cli

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Failed to install Railway CLI${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}🔐 Step 1: Login to Railway...${NC}"
railway login

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Railway login failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Logged in to Railway${NC}"
echo ""

# Deploy AI Models Service
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📦 SERVICE 1: AI Models (14 Models)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd python-services/ai-models

echo -e "${YELLOW}🏗️  Building AI Models service...${NC}"
echo ""

railway init --name "lydian-trader-ai-models"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Project already exists or init failed${NC}"
fi

# Set environment variables
echo -e "${YELLOW}🔧 Setting environment variables...${NC}"
railway variables set PORT=5003
railway variables set FLASK_ENV=production

echo ""
echo -e "${YELLOW}🚀 Deploying AI Models service...${NC}"
railway up

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ AI Models service deployed!${NC}"

    # Get deployment URL
    AI_MODELS_URL=$(railway domain 2>/dev/null)

    if [ -z "$AI_MODELS_URL" ]; then
        echo -e "${YELLOW}⚠️  Creating public domain...${NC}"
        railway domain
        AI_MODELS_URL=$(railway domain 2>/dev/null)
    fi

    echo -e "${GREEN}📡 AI Models URL: https://$AI_MODELS_URL${NC}"
    echo "$AI_MODELS_URL" > /tmp/ai_models_url.txt
else
    echo -e "${RED}❌ AI Models deployment failed${NC}"
    exit 1
fi

cd ../..
echo ""

# Deploy TA-Lib Service
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📦 SERVICE 2: TA-Lib (158 Indicators)${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

cd python-services/talib-service

echo -e "${YELLOW}🏗️  Building TA-Lib service...${NC}"
echo ""

railway init --name "lydian-trader-talib"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  Project already exists or init failed${NC}"
fi

# Set environment variables
echo -e "${YELLOW}🔧 Setting environment variables...${NC}"
railway variables set PORT=5005
railway variables set FLASK_ENV=production

echo ""
echo -e "${YELLOW}🚀 Deploying TA-Lib service...${NC}"
railway up

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ TA-Lib service deployed!${NC}"

    # Get deployment URL
    TALIB_URL=$(railway domain 2>/dev/null)

    if [ -z "$TALIB_URL" ]; then
        echo -e "${YELLOW}⚠️  Creating public domain...${NC}"
        railway domain
        TALIB_URL=$(railway domain 2>/dev/null)
    fi

    echo -e "${GREEN}📡 TA-Lib URL: https://$TALIB_URL${NC}"
    echo "$TALIB_URL" > /tmp/talib_url.txt
else
    echo -e "${RED}❌ TA-Lib deployment failed${NC}"
    exit 1
fi

cd ../..
echo ""

# Summary
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 DEPLOYMENT SUCCESSFUL!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ -f /tmp/ai_models_url.txt ] && [ -f /tmp/talib_url.txt ]; then
    AI_MODELS_URL=$(cat /tmp/ai_models_url.txt)
    TALIB_URL=$(cat /tmp/talib_url.txt)

    echo -e "${YELLOW}📋 Service URLs:${NC}"
    echo ""
    echo -e "  ${GREEN}AI Models Service:${NC}"
    echo -e "    https://$AI_MODELS_URL"
    echo ""
    echo -e "  ${GREEN}TA-Lib Service:${NC}"
    echo -e "    https://$TALIB_URL"
    echo ""
    echo -e "${YELLOW}🔧 Add these to Vercel environment variables:${NC}"
    echo ""
    echo "  NEXT_PUBLIC_AI_MODELS_URL=https://$AI_MODELS_URL"
    echo "  NEXT_PUBLIC_TALIB_SERVICE_URL=https://$TALIB_URL"
    echo ""

    # Save to .env.production
    echo "NEXT_PUBLIC_AI_MODELS_URL=https://$AI_MODELS_URL" > .env.production
    echo "NEXT_PUBLIC_TALIB_SERVICE_URL=https://$TALIB_URL" >> .env.production
    echo "NEXT_PUBLIC_BINANCE_API_URL=https://api.binance.com" >> .env.production
    echo "NEXT_PUBLIC_COINGECKO_API_URL=https://api.coingecko.com/api/v3" >> .env.production
    echo "NODE_ENV=production" >> .env.production
    echo "NEXT_TELEMETRY_DISABLED=1" >> .env.production

    echo -e "${GREEN}✅ Saved to .env.production${NC}"
    echo ""
fi

echo -e "${YELLOW}📊 Test your deployments:${NC}"
echo ""
echo "  AI Models Health:"
echo "    curl https://$AI_MODELS_URL/health"
echo ""
echo "  TA-Lib Health:"
echo "    curl https://$TALIB_URL/health"
echo ""

echo -e "${YELLOW}📝 Next steps:${NC}"
echo "  1. Copy environment variables to Vercel Dashboard"
echo "  2. Run: ./deploy-vercel.sh"
echo "  3. Test the full system"
echo ""

# Cleanup
rm -f /tmp/ai_models_url.txt /tmp/talib_url.txt
