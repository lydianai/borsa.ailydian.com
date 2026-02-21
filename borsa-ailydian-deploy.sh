#!/bin/bash

set -e

echo "============================================"
echo "🚀 BORSA.AILYDIAN.COM DEPLOYMENT SCRIPT"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if in correct directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}❌ package.json bulunamadı. Proje dizininde olduğunuzdan emin olun.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Proje dizini doğrulandı${NC}"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}⚠️  Vercel CLI kurulu değil. Kuruluyor...${NC}"
    npm install -g vercel
    echo -e "${GREEN}✅ Vercel CLI kuruldu${NC}"
else
    echo -e "${GREEN}✅ Vercel CLI zaten kurulu: $(vercel --version)${NC}"
fi
echo ""

# Login check
echo -e "${YELLOW}🔐 Vercel'e giriş yapılıyor...${NC}"
vercel whoami || vercel login
echo -e "${GREEN}✅ Vercel girişi başarılı${NC}"
echo ""

# Build production
echo -e "${YELLOW}🔨 Production build oluşturuluyor...${NC}"
pnpm build
echo -e "${GREEN}✅ Build başarılı${NC}"
echo ""

# Deploy to Vercel
echo -e "${YELLOW}📤 Vercel'e deploy ediliyor...${NC}"
vercel --prod
echo ""

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT TAMAMLANDI!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "${YELLOW}📝 SONRAKİ ADIMLAR:${NC}"
echo ""
echo "1. Vercel Dashboard'a gidin: https://vercel.com/dashboard"
echo "2. Projenizi seçin"
echo "3. Settings > Domains'e gidin"
echo "4. 'borsa.ailydian.com' domain'ini ekleyin"
echo "5. DNS ayarlarınızı güncelleyin (Vercel size talimatları gösterecek)"
echo ""
echo "6. Settings > Environment Variables'a gidin ve şu değişkenleri ekleyin:"
echo "   - NEXT_PUBLIC_PERSONAL_AUTH_ENABLED=0"
echo "   - NEXT_PUBLIC_FREEZE_TIME_TO=2025-10-27T10:00:00+03:00"
echo "   - GROQ_API_KEY=<your_key>"
echo "   - TELEGRAM_BOT_TOKEN=<your_token>"
echo "   - TELEGRAM_ALLOWED_CHAT_IDS=<your_chat_ids>"
echo "   - NEXT_PUBLIC_APP_URL=https://borsa.ailydian.com"
echo "   - NODE_ENV=production"
echo ""
echo "7. Son olarak tekrar deploy edin: vercel --prod"
echo ""
echo -e "${GREEN}🎉 Tebrikler! Sisteminiz live olacak!${NC}"
