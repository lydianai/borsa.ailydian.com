#!/bin/bash
set -e

source /tmp/azure-credentials.env

echo "🚀 Azure Services Setup Başlıyor..."
echo "Resource Group: $AZURE_RESOURCE_GROUP"
echo "Region: $AZURE_REGION"

# Event Hub
EH_NAMESPACE="AilydianEventHubNS"
EH_NAME="BorsaStream"

echo ""
echo "📡 Event Hub oluşturuluyor..."
if ! az eventhubs namespace show -g "$AZURE_RESOURCE_GROUP" -n "$EH_NAMESPACE" >/dev/null 2>&1; then
  echo "✨ Event Hub Namespace oluşturuluyor: $EH_NAMESPACE"
  az eventhubs namespace create \
    -g "$AZURE_RESOURCE_GROUP" \
    -n "$EH_NAMESPACE" \
    -l "$AZURE_REGION" \
    --sku Standard \
    --output none
else
  echo "✅ Event Hub Namespace mevcut: $EH_NAMESPACE"
fi

if ! az eventhubs eventhub show -g "$AZURE_RESOURCE_GROUP" --namespace-name "$EH_NAMESPACE" -n "$EH_NAME" >/dev/null 2>&1; then
  echo "✨ Event Hub oluşturuluyor: $EH_NAME"
  az eventhubs eventhub create \
    -g "$AZURE_RESOURCE_GROUP" \
    --namespace-name "$EH_NAMESPACE" \
    -n "$EH_NAME" \
    --message-retention 1 \
    --partition-count 2 \
    --output none
else
  echo "✅ Event Hub mevcut: $EH_NAME"
fi

echo "🔑 Event Hub connection string alınıyor..."
EH_CONN=$(az eventhubs namespace authorization-rule keys list \
  -g "$AZURE_RESOURCE_GROUP" \
  --namespace-name "$EH_NAMESPACE" \
  -n RootManageSharedAccessKey \
  --query primaryConnectionString -o tsv)

echo "✅ Event Hub hazır!"

# SignalR
SIGNALR_NAME="BorsaSignalR"

echo ""
echo "📡 SignalR Service oluşturuluyor..."
if ! az signalr show -g "$AZURE_RESOURCE_GROUP" -n "$SIGNALR_NAME" >/dev/null 2>&1; then
  echo "✨ SignalR Service oluşturuluyor: $SIGNALR_NAME (bu 2-3 dakika sürebilir...)"
  az signalr create \
    -g "$AZURE_RESOURCE_GROUP" \
    -n "$SIGNALR_NAME" \
    -l "$AZURE_REGION" \
    --sku Standard_S1 \
    --unit-count 1 \
    --output none
else
  echo "✅ SignalR Service mevcut: $SIGNALR_NAME"
fi

echo "🔑 SignalR connection string alınıyor..."
SIGNALR_CONN=$(az signalr key list \
  -g "$AZURE_RESOURCE_GROUP" \
  -n "$SIGNALR_NAME" \
  --query primaryConnectionString -o tsv)

SIGNALR_HOST=$(az signalr show \
  -g "$AZURE_RESOURCE_GROUP" \
  -n "$SIGNALR_NAME" \
  --query hostName -o tsv)

echo "✅ SignalR Service hazır!"

# Save all credentials
cat > /tmp/azure-services.env << EOF
# Event Hub
AZURE_EVENTHUB_NAMESPACE=$EH_NAMESPACE
AZURE_EVENTHUB_NAME=$EH_NAME
AZURE_EVENTHUB_CONN=$EH_CONN

# SignalR
AZURE_SIGNALR_NAME=$SIGNALR_NAME
AZURE_SIGNALR_CONN=$SIGNALR_CONN
AZURE_SIGNALR_HOST=$SIGNALR_HOST
EOF

echo ""
echo "✅ Azure Services hazır!"
echo "📄 Credentials: /tmp/azure-services.env"
