#!/bin/bash

echo "🔍 REVERSE ENGINEERING TEST - Gerçek vs Sistem Fiyatları"
echo "========================================================"
echo ""

# Test BTC
echo "📊 BITCOIN (BTCUSDT):"
echo "--------------------"
echo -n "Binance Real:  "
curl -s "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=BTCUSDT" | jq -r '"$" + .lastPrice + " (24h: " + .priceChangePercent + "%)"'

echo -n "System Price:  "
curl -s "http://localhost:3000/api/binance/futures" | jq -r '.data.all[] | select(.symbol == "BTCUSDT") | "$" + (.price|tostring) + " (24h: " + (.changePercent24h|tostring) + "%)"'
echo ""

# Test ETH
echo "📊 ETHEREUM (ETHUSDT):"
echo "--------------------"
echo -n "Binance Real:  "
curl -s "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=ETHUSDT" | jq -r '"$" + .lastPrice + " (24h: " + .priceChangePercent + "%)"'

echo -n "System Price:  "
curl -s "http://localhost:3000/api/binance/futures" | jq -r '.data.all[] | select(.symbol == "ETHUSDT") | "$" + (.price|tostring) + " (24h: " + (.changePercent24h|tostring) + "%)"'
echo ""

# Test SOL
echo "📊 SOLANA (SOLUSDT):"
echo "--------------------"
echo -n "Binance Real:  "
curl -s "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=SOLUSDT" | jq -r '"$" + .lastPrice + " (24h: " + .priceChangePercent + "%)"'

echo -n "System Price:  "
curl -s "http://localhost:3000/api/binance/futures" | jq -r '.data.all[] | select(.symbol == "SOLUSDT") | "$" + (.price|tostring) + " (24h: " + (.changePercent24h|tostring) + "%)"'
echo ""

# Test random low cap
echo "📊 LEVER (LEVERUSDT) - Low Cap Test:"
echo "--------------------"
echo -n "Binance Real:  "
curl -s "https://fapi.binance.com/fapi/v1/ticker/24hr?symbol=LEVERUSDT" | jq -r '"$" + .lastPrice + " (24h: " + .priceChangePercent + "%)"'

echo -n "System Price:  "
curl -s "http://localhost:3000/api/binance/futures" | jq -r '.data.all[] | select(.symbol == "LEVERUSDT") | "$" + (.price|tostring) + " (24h: " + (.changePercent24h|tostring) + "%)"'
echo ""

echo "========================================================"
echo "✅ Karşılaştırma tamamlandı!"
