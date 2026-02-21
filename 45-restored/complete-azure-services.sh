#!/bin/bash
set -e

source .env.azure

echo "🚀 Azure Services - Final Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Event Hub Connection String
echo "🔑 Event Hub connection string..."
EVENTHUB_CONN=$(az eventhubs namespace authorization-rule keys list \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --namespace-name "$AZURE_EVENTHUB_NAMESPACE" \
  --name RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)
echo "✅ Event Hub connection: ${EVENTHUB_CONN:0:50}..."

# 2. SignalR Service
echo ""
echo "📡 SignalR Service oluşturuluyor..."
if ! az signalr show -g "$AZURE_RESOURCE_GROUP" -n "$AZURE_SIGNALR_NAME" >/dev/null 2>&1; then
  echo "   ✨ SignalR oluşturuluyor (2-3 dakika)..."
  az signalr create \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --name "$AZURE_SIGNALR_NAME" \
    --location "$AZURE_REGION" \
    --sku Standard_S1 \
    --unit-count 1 \
    --service-mode Default \
    --output none
  echo "   ✅ SignalR oluşturuldu"
else
  echo "   ✅ SignalR mevcut"
fi

echo "🔑 SignalR connection string..."
SIGNALR_CONN=$(az signalr key list \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$AZURE_SIGNALR_NAME" \
  --query primaryConnectionString -o tsv)

SIGNALR_HOST=$(az signalr show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --name "$AZURE_SIGNALR_NAME" \
  --query hostName -o tsv)

echo "✅ SignalR connection: ${SIGNALR_CONN:0:50}..."
echo "✅ SignalR host: $SIGNALR_HOST"

# 3. Update .env.azure
echo ""
echo "💾 .env.azure güncelleniyor..."
cat > .env.azure << EOF
# ============================================
# AZURE CREDENTIALS - Ailydian + Borsa
# Generated: $(date +%Y-%m-%d)  
# ACTIVE & TESTED ✅
# ============================================

# Azure Core (MyAilydianApp)
AZURE_TENANT_ID=$AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID=$AZURE_SUBSCRIPTION_ID
AZURE_CLIENT_ID=$AZURE_CLIENT_ID
AZURE_CLIENT_SECRET=$AZURE_CLIENT_SECRET
AZURE_RESOURCE_GROUP=$AZURE_RESOURCE_GROUP
AZURE_REGION=$AZURE_REGION
AZURE_APP_NAME=$AZURE_APP_NAME

# Event Hub (ACTIVE)
AZURE_EVENTHUB_NAMESPACE=$AZURE_EVENTHUB_NAMESPACE
AZURE_EVENTHUB_NAME=$AZURE_EVENTHUB_NAME
AZURE_EVENTHUB_CONN=$EVENTHUB_CONN

# SignalR (ACTIVE)
AZURE_SIGNALR_NAME=$AZURE_SIGNALR_NAME
AZURE_SIGNALR_CONN=$SIGNALR_CONN
AZURE_SIGNALR_HOST=$SIGNALR_HOST

# Azure OpenAI (Optional - Add if available)
# AZURE_OPENAI_ENDPOINT=
# AZURE_OPENAI_KEY=
# AZURE_OPENAI_DEPLOYMENT_NAME=

# Borsa Microservices Endpoints
BORSA_MARKET_API=https://api.borsa.ailydian.com
BORSA_TRADING_ENDPOINT=https://api.borsa.ailydian.com/trading
BORSA_SIGNAL_ENDPOINT=https://api.borsa.ailydian.com/signals
EOF

echo "✅ .env.azure güncellendi!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Azure Services Hazır!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Event Hub: $AZURE_EVENTHUB_NAMESPACE/$AZURE_EVENTHUB_NAME"
echo "✅ SignalR: $SIGNALR_HOST"
echo "✅ Resource Group: $AZURE_RESOURCE_GROUP"
echo ""
echo "📄 Credentials: ~/Desktop/borsa/.env.azure"
