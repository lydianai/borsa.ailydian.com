#!/bin/bash
set -e

echo "🚀 Azure Complete Setup - Event Hub + SignalR + Functions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Load credentials
if [ ! -f .env.azure ]; then
  echo "❌ .env.azure bulunamadı!"
  exit 1
fi

export $(cat .env.azure | grep -v '^#' | xargs)

echo "✅ Credentials yüklendi"
echo "   Subscription: $AZURE_SUBSCRIPTION_ID"
echo "   Resource Group: $AZURE_RESOURCE_GROUP"
echo "   Region: $AZURE_REGION"
echo ""

# Set subscription
az account set --subscription $AZURE_SUBSCRIPTION_ID

# 1. Event Hub
echo "📡 Event Hub kurulumu..."
if ! az eventhubs namespace show -g "$AZURE_RESOURCE_GROUP" -n "$AZURE_EVENTHUB_NAMESPACE" >/dev/null 2>&1; then
  echo "   ✨ Event Hub Namespace oluşturuluyor: $AZURE_EVENTHUB_NAMESPACE"
  az eventhubs namespace create \
    -g "$AZURE_RESOURCE_GROUP" \
    -n "$AZURE_EVENTHUB_NAMESPACE" \
    -l "$AZURE_REGION" \
    --sku Standard \
    --output none
  echo "   ✅ Namespace oluşturuldu"
else
  echo "   ✅ Namespace mevcut: $AZURE_EVENTHUB_NAMESPACE"
fi

if ! az eventhubs eventhub show -g "$AZURE_RESOURCE_GROUP" --namespace-name "$AZURE_EVENTHUB_NAMESPACE" -n "$AZURE_EVENTHUB_NAME" >/dev/null 2>&1; then
  echo "   ✨ Event Hub oluşturuluyor: $AZURE_EVENTHUB_NAME"
  az eventhubs eventhub create \
    -g "$AZURE_RESOURCE_GROUP" \
    --namespace-name "$AZURE_EVENTHUB_NAMESPACE" \
    -n "$AZURE_EVENTHUB_NAME" \
    --retention-time-in-hours 1 \
    --partition-count 2 \
    --output none
  echo "   ✅ Event Hub oluşturuldu"
else
  echo "   ✅ Event Hub mevcut: $AZURE_EVENTHUB_NAME"
fi

echo "   🔑 Connection string alınıyor..."
EVENTHUB_CONN=$(az eventhubs namespace authorization-rule keys list \
  -g "$AZURE_RESOURCE_GROUP" \
  --namespace-name "$AZURE_EVENTHUB_NAMESPACE" \
  -n RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)
echo "   ✅ Connection string alındı"

# 2. SignalR
echo ""
echo "📡 SignalR Service kurulumu..."
if ! az signalr show -g "$AZURE_RESOURCE_GROUP" -n "$AZURE_SIGNALR_NAME" >/dev/null 2>&1; then
  echo "   ✨ SignalR oluşturuluyor: $AZURE_SIGNALR_NAME (2-3 dakika sürebilir...)"
  az signalr create \
    -g "$AZURE_RESOURCE_GROUP" \
    -n "$AZURE_SIGNALR_NAME" \
    -l "$AZURE_REGION" \
    --sku Standard_S1 \
    --unit-count 1 \
    --service-mode Default \
    --output none
  echo "   ✅ SignalR oluşturuldu"
else
  echo "   ✅ SignalR mevcut: $AZURE_SIGNALR_NAME"
fi

echo "   🔑 SignalR connection string alınıyor..."
SIGNALR_CONN=$(az signalr key list \
  -g "$AZURE_RESOURCE_GROUP" \
  -n "$AZURE_SIGNALR_NAME" \
  --query primaryConnectionString -o tsv)

SIGNALR_HOST=$(az signalr show \
  -g "$AZURE_RESOURCE_GROUP" \
  -n "$AZURE_SIGNALR_NAME" \
  --query hostName -o tsv)

echo "   ✅ SignalR connection string alındı"
echo "   📍 SignalR Host: $SIGNALR_HOST"

# Save updated .env.azure
echo ""
echo "💾 .env.azure güncelleniyor..."
cat > .env.azure.updated << EOF
# ============================================
# AZURE CREDENTIALS - Ailydian + Borsa
# Generated: $(date +%Y-%m-%d)
# ============================================

# Azure Core (MyAilydianApp)
AZURE_TENANT_ID=$AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID=$AZURE_SUBSCRIPTION_ID
AZURE_CLIENT_ID=$AZURE_CLIENT_ID
AZURE_CLIENT_SECRET=$AZURE_CLIENT_SECRET
AZURE_RESOURCE_GROUP=$AZURE_RESOURCE_GROUP
AZURE_REGION=$AZURE_REGION
AZURE_APP_NAME=$AZURE_APP_NAME

# Event Hub
AZURE_EVENTHUB_NAMESPACE=$AZURE_EVENTHUB_NAMESPACE
AZURE_EVENTHUB_NAME=$AZURE_EVENTHUB_NAME
AZURE_EVENTHUB_CONN=$EVENTHUB_CONN

# SignalR
AZURE_SIGNALR_NAME=$AZURE_SIGNALR_NAME
AZURE_SIGNALR_CONN=$SIGNALR_CONN
AZURE_SIGNALR_HOST=$SIGNALR_HOST

# Azure OpenAI (if exists)
# AZURE_OPENAI_ENDPOINT=
# AZURE_OPENAI_KEY=
# AZURE_OPENAI_DEPLOYMENT_NAME=

# Borsa Microservices Endpoints
BORSA_MARKET_API=https://api.borsa.ailydian.com
BORSA_TRADING_ENDPOINT=https://api.borsa.ailydian.com/trading
BORSA_SIGNAL_ENDPOINT=https://api.borsa.ailydian.com/signals
EOF

mv .env.azure.updated .env.azure
echo "✅ .env.azure güncellendi!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Azure Setup Tamamlandı!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Event Hub: $AZURE_EVENTHUB_NAMESPACE/$AZURE_EVENTHUB_NAME"
echo "✅ SignalR: $SIGNALR_HOST"
echo "✅ Resource Group: $AZURE_RESOURCE_GROUP"
echo ""
echo "📄 Credentials: .env.azure"
