#!/bin/bash
# Ngrok ile Rate Limiting Entegrasyonu
# Groq API için rate limiting ile production-like environment

echo "🚀 Starting Ngrok with Rate Limiting..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Rate Limit Configuration:"
echo "   • Global API: 100 requests/minute per IP"
echo "   • Groq API:   30 requests/minute per IP (Free tier)"
echo "   • Auth API:   20 requests/minute per IP"
echo "   • Market:     200 requests/minute per IP"
echo ""
echo "⚠️  Make sure your app is running on port 3000"
echo "   Run: pnpm dev"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Error: ngrok is not installed"
    echo "   Install: https://ngrok.com/download"
    exit 1
fi

# Check if rate-limit.yml exists
if [ ! -f "rate-limit.yml" ]; then
    echo "❌ Error: rate-limit.yml not found"
    exit 1
fi

# Get ngrok domain (replace with your custom domain if you have one)
NGROK_DOMAIN="${NGROK_DOMAIN:-}"

if [ -z "$NGROK_DOMAIN" ]; then
    echo "📝 Starting ngrok with random domain..."
    echo "   To use custom domain: export NGROK_DOMAIN=your-domain.ngrok.app"
    echo ""
    ngrok http 3000 --traffic-policy-file rate-limit.yml
else
    echo "🌐 Starting ngrok with custom domain: $NGROK_DOMAIN"
    echo ""
    ngrok http 3000 --domain "$NGROK_DOMAIN" --traffic-policy-file rate-limit.yml
fi
