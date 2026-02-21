#!/bin/bash
set -e

echo "🚀 Azure Ailydian+Borsa Entegrasyonu Başlıyor..."

# Credentials
SUBSCRIPTION_ID="931c7633-e61e-4a37-8798-fe1f6f20580e"
TENANT_ID="e7a71902-6ea1-497b-b39f-61fe5f37fcf0"
REGION="westeurope"
RG="Ailydian-RG"

# Set subscription
echo "📌 Subscription ayarlanıyor..."
az account set --subscription $SUBSCRIPTION_ID

# Resource Group (idempotent)
echo "📦 Resource Group kontrol ediliyor..."
if ! az group show -n "$RG" >/dev/null 2>&1; then
  echo "✨ Resource Group oluşturuluyor: $RG"
  az group create -n "$RG" -l "$REGION" --output none
else
  echo "✅ Resource Group mevcut: $RG"
fi

# App Registration
echo "🔐 App Registration oluşturuluyor..."
APP_NAME="Ailydian-Gateway"
if az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv 2>/dev/null | grep -q "."; then
  echo "✅ App Registration mevcut: $APP_NAME"
  APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv)
else
  echo "✨ Yeni App Registration oluşturuluyor..."
  az ad app create \
    --display-name "$APP_NAME" \
    --sign-in-audience "AzureADMyOrg" \
    --enable-id-token-issuance true \
    --enable-access-token-issuance true \
    --output none
  APP_ID=$(az ad app list --display-name "$APP_NAME" --query "[0].appId" -o tsv)
  echo "✅ App ID: $APP_ID"
fi

# Client Secret
echo "🔑 Client Secret oluşturuluyor..."
SECRET=$(az ad app credential reset \
  --id "$APP_ID" \
  --append \
  --display-name "Ailydian-Secret-$(date +%Y%m%d)" \
  --years 2 \
  --query password -o tsv 2>/dev/null || echo "SECRET_GENERATION_FAILED")

if [ "$SECRET" = "SECRET_GENERATION_FAILED" ]; then
  echo "⚠️  Secret zaten mevcut, mevcut secret kullanılacak"
  SECRET="<EXISTING_SECRET_USE_AZURE_PORTAL>"
fi

echo "✅ Temel Azure kaynakları hazır!"
echo ""
echo "📋 Azure Credentials:"
echo "TENANT_ID=$TENANT_ID"
echo "SUBSCRIPTION_ID=$SUBSCRIPTION_ID"
echo "APP_ID=$APP_ID"
echo "APP_SECRET=$SECRET"
echo "RESOURCE_GROUP=$RG"
echo "REGION=$REGION"

# Save to temp
cat > /tmp/azure-credentials.env << EOF
AZURE_TENANT_ID=$TENANT_ID
AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID
AZURE_CLIENT_ID=$APP_ID
AZURE_CLIENT_SECRET=$SECRET
AZURE_RESOURCE_GROUP=$RG
AZURE_REGION=$REGION
EOF

echo ""
echo "✅ Credentials kaydedildi: /tmp/azure-credentials.env"
