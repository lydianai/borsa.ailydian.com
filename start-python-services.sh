#!/bin/bash

###############################################################################
# 🚀 PYTHON SERVICES STARTUP SCRIPT
#
# Bu script tüm kritik Python mikroservislerini başlatır
# Binance API timeout sorunlarını çözmek için tasarlanmıştır
#
# KULLANIM:
#   chmod +x start-python-services.sh
#   ./start-python-services.sh
#
# Servisler:
#   - WebSocket Streaming (Port 5021) - Gerçek zamanlı fiyatlar
#   - TA-Lib Service (Port 5002) - Teknik analiz
#   - AI Models (Port 5003) - AI tahminleri
#   - Quantum Ladder (Port 5022) - Fibonacci analizi
###############################################################################

set -e  # Exit on error

# Renkler
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="/Users/sardag/Documents/ailydian-signal"
PYTHON_SERVICE_DIR="$PROJECT_ROOT/Phyton-Service"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🚀 PYTHON SERVİSLERİNİ BAŞLATILIYOR${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Önceki servisleri temizle
echo -e "${YELLOW}🧹 Eski servisleri temizleniyor...${NC}"
for port in 5002 5003 5021 5022; do
    if lsof -ti:$port > /dev/null 2>&1; then
        echo -e "  ${RED}✗${NC} Port $port kullanımda, durduruli yor..."
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
done
echo -e "${GREEN}✓ Portlar temizlendi${NC}\n"

# Service 1: WebSocket Streaming (En yüksek öncelik - gerçek zamanlı veri)
echo -e "${YELLOW}📡 WebSocket Streaming Service başlatılıyor (Port 5021)...${NC}"
cd "$PYTHON_SERVICE_DIR/websocket-streaming"
if [ -d "venv" ]; then
    nohup ./venv/bin/python3 app.py > logs/service.log 2>&1 &
    echo $! > .pid
    sleep 3

    if lsof -ti:5021 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ WebSocket Streaming çalışıyor${NC}"
    else
        echo -e "${RED}✗ WebSocket Streaming başlatılamadı${NC}"
        echo -e "${YELLOW}  Log: $PYTHON_SERVICE_DIR/websocket-streaming/logs/service.log${NC}"
    fi
else
    echo -e "${RED}✗ venv bulunamadı, servis atlanıyor${NC}"
fi
echo ""

# Service 2: TA-Lib (Teknik analiz)
echo -e "${YELLOW}📊 TA-Lib Service başlatılıyor (Port 5002)...${NC}"
cd "$PYTHON_SERVICE_DIR/talib-service"
if [ -d "venv" ]; then
    nohup ./venv/bin/python3 app.py > logs/service.log 2>&1 &
    echo $! > .pid
    sleep 3

    if lsof -ti:5002 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ TA-Lib Service çalışıyor${NC}"
    else
        echo -e "${RED}✗ TA-Lib Service başlatılamadı${NC}"
        echo -e "${YELLOW}  Log: $PYTHON_SERVICE_DIR/talib-service/logs/service.log${NC}"
    fi
else
    echo -e "${RED}✗ venv bulunamadı, servis atlanıyor${NC}"
fi
echo ""

# Service 3: AI Models
echo -e "${YELLOW}🤖 AI Models Service başlatılıyor (Port 5003)...${NC}"
cd "$PYTHON_SERVICE_DIR/ai-models"
if [ -d "venv" ]; then
    nohup ./venv/bin/python3 app.py > logs/service.log 2>&1 &
    echo $! > .pid
    sleep 3

    if lsof -ti:5003 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ AI Models Service çalışıyor${NC}"
    else
        echo -e "${RED}✗ AI Models Service başlatılamadı${NC}"
        echo -e "${YELLOW}  Log: $PYTHON_SERVICE_DIR/ai-models/logs/service.log${NC}"
    fi
else
    echo -e "${RED}✗ venv bulunamadı, servis atlanıyor${NC}"
fi
echo ""

# Service 4: Quantum Ladder
echo -e "${YELLOW}🔮 Quantum Ladder Service başlatılıyor (Port 5022)...${NC}"
cd "$PYTHON_SERVICE_DIR/quantum-ladder"
if [ -d "venv" ]; then
    nohup ./venv/bin/python3 app.py > logs/service.log 2>&1 &
    echo $! > .pid
    sleep 3

    if lsof -ti:5022 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Quantum Ladder Service çalışıyor${NC}"
    else
        echo -e "${RED}✗ Quantum Ladder Service başlatılamadı${NC}"
        echo -e "${YELLOW}  Log: $PYTHON_SERVICE_DIR/quantum-ladder/logs/service.log${NC}"
    fi
else
    echo -e "${RED}✗ venv bulunamadı, servis atlanıyor${NC}"
fi
echo ""

# Servis durumunu kontrol et
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📊 SERVİS DURUMU${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

RUNNING=0
TOTAL=4

for port in 5021:WebSocket 5002:TA-Lib 5003:AI-Models 5022:Quantum; do
    IFS=':' read -ra ADDR <<< "$port"
    PORT_NUM="${ADDR[0]}"
    SERVICE_NAME="${ADDR[1]}"

    if lsof -ti:$PORT_NUM > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $SERVICE_NAME (Port $PORT_NUM)"
        ((RUNNING++))
    else
        echo -e "${RED}✗${NC} $SERVICE_NAME (Port $PORT_NUM) - DURDURULDU"
    fi
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}📈 TOPLAM: $RUNNING/$TOTAL servis çalışıyor${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Test endpoints
echo -e "${YELLOW}🧪 Endpointler test ediliyor...${NC}"
echo ""

# Test WebSocket
if lsof -ti:5021 > /dev/null 2>&1; then
    if curl -s http://localhost:5021/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} WebSocket /health endpoint çalışıyor"
    else
        echo -e "${YELLOW}⚠${NC}  WebSocket endpoint yanıt vermiyor"
    fi

    if curl -s http://localhost:5021/api/latest-prices > /dev/null 2>&1; then
        PRICE_COUNT=$(curl -s http://localhost:5021/api/latest-prices | grep -o '"count":[0-9]*' | cut -d: -f2)
        echo -e "${GREEN}✓${NC} WebSocket /api/latest-prices çalışıyor ($PRICE_COUNT fiyat)"
    else
        echo -e "${YELLOW}⚠${NC}  WebSocket /api/latest-prices yanıt vermiyor"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ PYTHON SERVİSLERİ BAŞLATILDI${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}📝 İPUCU:${NC}"
echo -e "  - Servisleri durdurmak için: ${GREEN}./stop-python-services.sh${NC}"
echo -e "  - Logları görmek için: ${GREEN}tail -f Phyton-Service/*/logs/service.log${NC}"
echo -e "  - Sistem durumu: ${GREEN}http://localhost:5021/stats${NC}"
echo ""
