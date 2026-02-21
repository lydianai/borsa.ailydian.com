#!/bin/bash
##
# 🚀 TELEGRAM SCHEDULER AUTO-START SCRIPT
# Bu script tüm Telegram servislerini otomatik başlatır
##

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 TELEGRAM SCHEDULER BAŞLATILIYOR..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Working directory
cd /Users/sardag/Documents/sardag-emrah-final.bak-20251030-170900/Telegram/schedulers

# PM2 kontrol
if ! command -v pm2 &> /dev/null; then
    echo "❌ PM2 bulunamadı! npm install -g pm2 ile yükleyin."
    exit 1
fi

# Eski PM2 process'lerini temizle
echo "🧹 Eski PM2 process'leri temizleniyor..."
pm2 delete telegram-scheduler 2>/dev/null || true
pm2 delete telegram-bot 2>/dev/null || true

# Yeni PM2 process'lerini başlat
echo "🚀 PM2 servisleri başlatılıyor..."
pm2 start ecosystem.config.js

# Otomatik başlatma (bilgisayar restart)
echo "💾 PM2 startup ayarlanıyor..."
pm2 startup | grep -o 'sudo.*' | bash || true
pm2 save

# Status göster
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ TELEGRAM SCHEDULER BAŞLATILDI!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
pm2 list

echo ""
echo "📊 KULLANIM:"
echo "   pm2 logs telegram-scheduler  # Log'ları izle"
echo "   pm2 monit                    # Monitoring"
echo "   pm2 restart telegram-scheduler # Restart"
echo ""
