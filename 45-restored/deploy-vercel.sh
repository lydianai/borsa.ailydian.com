#!/bin/bash

# Vercel Deployment Script for Lydian Trader
# Author: Claude Code
# Date: 2025-10-02

echo "🚀 VERCEL DEPLOYMENT - Lydian Trader"
echo "===================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}❌ Vercel CLI not found${NC}"
    echo "Installing Vercel CLI..."
    npm install -g vercel
fi

echo -e "${YELLOW}📦 Step 1: Installing dependencies...${NC}"
npm install

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ npm install failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

echo -e "${YELLOW}🔍 Step 2: Type checking...${NC}"
npm run type-check

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Type check failed${NC}"
    echo "Fix TypeScript errors before deploying"
    exit 1
fi

echo -e "${GREEN}✅ Type check passed${NC}"
echo ""

echo -e "${YELLOW}🏗️  Step 3: Building for production...${NC}"
npm run build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build successful${NC}"
echo ""

echo -e "${YELLOW}⚠️  Important: Make sure you have set these environment variables in Vercel:${NC}"
echo ""
echo "  NEXT_PUBLIC_AI_MODELS_URL=https://your-ai-models.up.railway.app"
echo "  NEXT_PUBLIC_TALIB_SERVICE_URL=https://your-talib-service.up.railway.app"
echo "  NEXT_PUBLIC_BINANCE_API_URL=https://api.binance.com"
echo "  NEXT_PUBLIC_COINGECKO_API_URL=https://api.coingecko.com/api/v3"
echo "  NODE_ENV=production"
echo "  NEXT_TELEMETRY_DISABLED=1"
echo ""
echo "  Optional (Azure OpenAI):"
echo "  AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com"
echo "  AZURE_OPENAI_API_KEY=your_api_key"
echo "  AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4"
echo ""

read -p "Have you configured all environment variables? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠️  Please configure environment variables in Vercel Dashboard first${NC}"
    echo "   Vercel Dashboard → Project → Settings → Environment Variables"
    exit 1
fi

echo ""
echo -e "${YELLOW}🚀 Step 4: Deploying to Vercel...${NC}"
echo ""

# Deploy to production
vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ DEPLOYMENT SUCCESSFUL!${NC}"
    echo ""
    echo "🎉 Your app is now live on Vercel!"
    echo ""
    echo "Next steps:"
    echo "  1. Test the deployment: vercel inspect"
    echo "  2. View logs: vercel logs"
    echo "  3. Open your app: vercel open"
    echo ""
else
    echo -e "${RED}❌ Deployment failed${NC}"
    echo "Check Vercel logs for details"
    exit 1
fi
