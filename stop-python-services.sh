#!/bin/bash

###############################################################################
# 🛑 PYTHON SERVICES STOP SCRIPT
#
# Tüm Python mikroservislerini güvenli şekilde durdurur
###############################################################################

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}🛑 PYTHON SERVİSLERİ DURDURULUYOR${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

for port in 5021:WebSocket 5002:TA-Lib 5003:AI-Models 5022:Quantum; do
    IFS=':' read -ra ADDR <<< "$port"
    PORT_NUM="${ADDR[0]}"
    SERVICE_NAME="${ADDR[1]}"

    if lsof -ti:$PORT_NUM > /dev/null 2>&1; then
        echo -e "${YELLOW}Durduruluyor: $SERVICE_NAME (Port $PORT_NUM)${NC}"
        lsof -ti:$PORT_NUM | xargs kill -9 2>/dev/null || true
        echo -e "${GREEN}✓${NC} $SERVICE_NAME durduruldu"
    else
        echo -e "${YELLOW}⚠${NC}  $SERVICE_NAME zaten durdurulmuş"
    fi
done

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ TÜM SERVİSLER DURDURULDU${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
