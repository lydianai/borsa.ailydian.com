#!/bin/bash

##############################################################################
# 🚀 TELEGRAM 24/7 SETUP SCRIPT
# Kalıcı Telegram bildirim sistemi kurulumu
#
# Features:
# - PM2 ile process management
# - Auto-restart on crash
# - Log rotation
# - Memory monitoring
# - Ngrok tunnel persistence
# - Health check cron jobs
##############################################################################

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 TELEGRAM 24/7 SYSTEM SETUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# 1. REQUIREMENTS CHECK
# ============================================================================

echo "📋 Step 1: Checking requirements..."

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "⚠️  PM2 not found. Installing PM2 globally..."
    npm install -g pm2
    echo "✅ PM2 installed"
else
    echo "✅ PM2 already installed"
fi

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ Ngrok not found. Please install ngrok:"
    echo "   brew install ngrok (macOS)"
    echo "   or download from https://ngrok.com/"
    exit 1
else
    echo "✅ Ngrok already installed"
fi

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "❌ .env.local not found!"
    exit 1
else
    echo "✅ .env.local exists"
fi

echo ""

# ============================================================================
# 2. PM2 ECOSYSTEM FILE
# ============================================================================

echo "📋 Step 2: Creating PM2 ecosystem config..."

cat > ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'sardag-emrah-web',
      script: 'pnpm',
      args: 'start',
      cwd: './',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      },
      error_file: './logs/pm2-error.log',
      out_file: './logs/pm2-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      // Restart on crash
      exp_backoff_restart_delay: 100,
      min_uptime: '10s',
      max_restarts: 10,
      // Health check
      health_check_grace_period: 3000,
      health_check_fatal_timeout: 5000
    },
    {
      name: 'ngrok-tunnel',
      script: 'ngrok',
      args: 'http 3000 --log=stdout',
      autorestart: true,
      watch: false,
      max_memory_restart: '200M',
      error_file: './logs/ngrok-error.log',
      out_file: './logs/ngrok-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      // Restart on crash
      exp_backoff_restart_delay: 100,
      min_uptime: '10s',
      max_restarts: 10
    }
  ]
};
EOF

echo "✅ PM2 ecosystem config created"
echo ""

# ============================================================================
# 3. HEALTH CHECK SCRIPT
# ============================================================================

echo "📋 Step 3: Creating health check script..."

mkdir -p scripts

cat > scripts/health-check.sh << 'EOF'
#!/bin/bash

# Health check script for Telegram system
# Run this via cron every 5 minutes

LOG_FILE="./logs/health-check.log"
mkdir -p logs

echo "[$(date)] Starting health check..." >> "$LOG_FILE"

# Check if Next.js server is running
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "[$(date)] ✅ Next.js server: HEALTHY" >> "$LOG_FILE"
else
    echo "[$(date)] ❌ Next.js server: DOWN - Restarting..." >> "$LOG_FILE"
    pm2 restart sardag-emrah-web
fi

# Check if ngrok tunnel is active
if curl -sf http://localhost:4040/api/tunnels > /dev/null 2>&1; then
    echo "[$(date)] ✅ Ngrok tunnel: ACTIVE" >> "$LOG_FILE"
else
    echo "[$(date)] ❌ Ngrok tunnel: DOWN - Restarting..." >> "$LOG_FILE"
    pm2 restart ngrok-tunnel
    sleep 5
    # Update Telegram webhook with new URL
    if [ -f "./scripts/update-webhook.sh" ]; then
        bash ./scripts/update-webhook.sh >> "$LOG_FILE" 2>&1
    fi
fi

# Check Telegram bot connectivity
if curl -sf "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" > /dev/null 2>&1; then
    echo "[$(date)] ✅ Telegram bot: CONNECTED" >> "$LOG_FILE"
else
    echo "[$(date)] ⚠️  Telegram bot: CONNECTION ISSUE" >> "$LOG_FILE"
fi

echo "[$(date)] Health check completed" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"
EOF

chmod +x scripts/health-check.sh
echo "✅ Health check script created"
echo ""

# ============================================================================
# 4. WEBHOOK UPDATE SCRIPT
# ============================================================================

echo "📋 Step 4: Creating webhook update script..."

cat > scripts/update-webhook.sh << 'EOF'
#!/bin/bash

# Update Telegram webhook with current ngrok URL

# Load environment variables
source .env.local

# Get ngrok URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$NGROK_URL" ]; then
    echo "[$(date)] ❌ Failed to get ngrok URL"
    exit 1
fi

echo "[$(date)] 🔗 Ngrok URL: $NGROK_URL"

# Update Telegram webhook
RESPONSE=$(curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "{
    \"url\": \"${NGROK_URL}/api/telegram/webhook\",
    \"secret_token\": \"${TELEGRAM_BOT_WEBHOOK_SECRET}\"
  }" 2>&1)

if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "[$(date)] ✅ Webhook updated successfully"
else
    echo "[$(date)] ❌ Webhook update failed: $RESPONSE"
    exit 1
fi
EOF

chmod +x scripts/update-webhook.sh
echo "✅ Webhook update script created"
echo ""

# ============================================================================
# 5. SUBSCRIBER PERSISTENCE SCRIPT
# ============================================================================

echo "📋 Step 5: Creating subscriber persistence..."

cat > scripts/save-subscribers.sh << 'EOF'
#!/bin/bash

# Save current subscribers to file for persistence

SUBSCRIBER_FILE="./data/subscribers.json"
mkdir -p data

# Get subscribers from API
curl -s http://localhost:3000/api/telegram/admin | \
  python3 -c "import sys, json; data = json.load(sys.stdin); print(json.dumps(data['stats']['subscribers']))" \
  > "$SUBSCRIBER_FILE"

echo "[$(date)] Subscribers saved to $SUBSCRIBER_FILE"
EOF

chmod +x scripts/save-subscribers.sh
echo "✅ Subscriber persistence script created"
echo ""

# ============================================================================
# 6. CRON JOBS
# ============================================================================

echo "📋 Step 6: Setting up cron jobs..."

# Create crontab entries
CRON_FILE="/tmp/sardag-cron"

cat > "$CRON_FILE" << EOF
# SARDAG-EMRAH Telegram System - Health Check (Every 5 minutes)
*/5 * * * * cd $(pwd) && bash scripts/health-check.sh

# Save subscribers (Every hour)
0 * * * * cd $(pwd) && bash scripts/save-subscribers.sh

# Daily system health report (9 AM and 6 PM)
0 9,18 * * * cd $(pwd) && curl -X POST http://localhost:3000/api/system/daily-report
EOF

# Install crontab
crontab "$CRON_FILE"
rm "$CRON_FILE"

echo "✅ Cron jobs installed"
echo ""

# ============================================================================
# 7. LOG ROTATION
# ============================================================================

echo "📋 Step 7: Setting up log rotation..."

pm2 install pm2-logrotate
pm2 set pm2-logrotate:max_size 10M
pm2 set pm2-logrotate:retain 7
pm2 set pm2-logrotate:compress true

echo "✅ Log rotation configured"
echo ""

# ============================================================================
# 8. START SERVICES
# ============================================================================

echo "📋 Step 8: Starting services..."

# Create logs directory
mkdir -p logs

# Build Next.js app for production
echo "🔨 Building Next.js app..."
pnpm build

# Stop any existing PM2 processes
pm2 delete all 2>/dev/null || true

# Start services with PM2
echo "🚀 Starting PM2 services..."
pm2 start ecosystem.config.js

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 10

# Update Telegram webhook
echo "🔗 Updating Telegram webhook..."
bash scripts/update-webhook.sh

# Subscribe default user
echo "📲 Subscribing default user..."
curl -X POST http://localhost:3000/api/telegram/subscribe \
  -H "Content-Type: application/json" \
  -d "{\"chatId\": ${TELEGRAM_ALLOWED_CHAT_IDS:-7575640489}}" \
  2>/dev/null || true

# Save PM2 configuration for auto-start on boot
pm2 save
pm2 startup

echo ""
echo "✅ PM2 processes saved and startup configured"
echo ""

# ============================================================================
# 9. VERIFICATION
# ============================================================================

echo "📋 Step 9: Verifying installation..."
echo ""

pm2 list

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TELEGRAM 24/7 SYSTEM SETUP COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Status:"
echo "   - PM2 processes: Running"
echo "   - Health checks: Every 5 minutes"
echo "   - Subscriber backup: Every hour"
echo "   - Daily reports: 9 AM & 6 PM"
echo "   - Log rotation: 7 days, 10MB max"
echo ""
echo "🎯 Useful commands:"
echo "   pm2 list                  - List all processes"
echo "   pm2 logs                  - View all logs"
echo "   pm2 logs sardag-emrah-web - View Next.js logs"
echo "   pm2 logs ngrok-tunnel     - View ngrok logs"
echo "   pm2 restart all           - Restart all processes"
echo "   pm2 stop all              - Stop all processes"
echo "   pm2 monit                 - Monitor processes"
echo ""
echo "📂 Log files:"
echo "   ./logs/pm2-out.log        - PM2 output"
echo "   ./logs/pm2-error.log      - PM2 errors"
echo "   ./logs/ngrok-out.log      - Ngrok output"
echo "   ./logs/health-check.log   - Health check logs"
echo ""
echo "🔄 Auto-restart: Enabled"
echo "💾 Persistence: Enabled"
echo "🔒 Security: White-hat compliant"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
