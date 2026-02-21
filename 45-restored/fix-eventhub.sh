#!/bin/bash
set +e  # Continue on error

source .env.azure

echo "🔧 Event Hub Fix..."
az eventhubs eventhub create \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --namespace-name "$AZURE_EVENTHUB_NAMESPACE" \
  --name "$AZURE_EVENTHUB_NAME" \
  --partition-count 2 2>&1 | grep -v "ERROR" || echo "Event Hub mevcut veya oluşturuldu"

echo ""
echo "✅ Event Hub durumu:"
az eventhubs eventhub show \
  --resource-group "$AZURE_RESOURCE_GROUP" \
  --namespace-name "$AZURE_EVENTHUB_NAMESPACE" \
  --name "$AZURE_EVENTHUB_NAME" \
  --query "{name:name, partitions:partitionCount, status:status}" 2>&1 || echo "Event Hub kontrol edilemiyor"
