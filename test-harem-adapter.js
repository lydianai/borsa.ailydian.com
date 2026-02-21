/**
 * Test Harem Altın Adapter Parsing
 */

// Sample API response (real data from API)
const sampleAPIResponse = {
  "success": true,
  "message": "Query Success",
  "data": [
    {
      "key": "GRAM ALTIN",
      "buy": "5.855,92",
      "sell": "5.937,31",
      "percent": "1.64",
      "arrow": "up",
      "last_update": "25.10.2025 16:58:32"
    },
    {
      "key": "22 AYAR",
      "buy": "5.353,67",
      "sell": "5.603,57",
      "percent": "4.93",
      "arrow": "up",
      "last_update": "25.10.2025 16:58:32"
    },
    {
      "key": "GÜMÜŞ ONS",
      "buy": "48,55",
      "sell": "48,63",
      "percent": "0.16",
      "arrow": "up",
      "last_update": "25.10.2025 16:58:34"
    }
  ]
};

/**
 * Parse Turkish price format to number
 * Example: "5.855,92" -> 5855.92
 */
function parseTurkishPrice(priceStr) {
  if (!priceStr) return 0;
  // Remove thousand separators (dots) and replace decimal comma with dot
  const cleanPrice = priceStr.replace(/\./g, '').replace(',', '.');
  return parseFloat(cleanPrice) || 0;
}

/**
 * Parse gold price data from Harem Altın API response
 */
function parseGoldData(data) {
  const prices = [];

  try {
    if (data.success && Array.isArray(data.data)) {
      data.data.forEach((item) => {
        const key = item.key;
        const buyPrice = parseTurkishPrice(item.buy);
        const sellPrice = parseTurkishPrice(item.sell);
        const changePercent = parseFloat(item.percent) || 0;

        // Only include gold products (exclude silver, platinum, etc.)
        const goldKeywords = ['ALTIN', 'GRAM', 'ÇEYREK', 'YARIM', 'TAM', 'ATA', 'GREMSE', '14 AYAR', '22 AYAR', 'Has Altın'];
        const isGold = goldKeywords.some(keyword => key.includes(keyword));

        if (isGold && buyPrice > 0 && sellPrice > 0) {
          prices.push({
            symbol: key.toUpperCase().replace(/\s+/g, '_'),
            name: key,
            price: sellPrice, // Use sell price as main price
            change24h: changePercent,
            buyPrice: buyPrice,
            sellPrice: sellPrice,
            lastUpdate: new Date(),
            category: 'gold',
            currency: 'TRY',
          });
        }
      });
    }
  } catch (error) {
    console.error('Error parsing data:', error);
  }

  return prices;
}

// Run test
console.log('🧪 Testing Harem Altın Adapter Parsing...\n');

console.log('1️⃣ Testing parseTurkishPrice():');
console.log('   "5.855,92" →', parseTurkishPrice("5.855,92"), '(expected: 5855.92)');
console.log('   "5.937,31" →', parseTurkishPrice("5.937,31"), '(expected: 5937.31)');
console.log('   "48,55" →', parseTurkishPrice("48,55"), '(expected: 48.55)');

console.log('\n2️⃣ Testing parseGoldData():');
const parsed = parseGoldData(sampleAPIResponse);
console.log(`   Found ${parsed.length} gold products (expected: 2, GÜMÜŞ should be filtered out)`);

parsed.forEach((product, i) => {
  console.log(`\n   Product ${i + 1}:`);
  console.log(`   - Symbol: ${product.symbol}`);
  console.log(`   - Name: ${product.name}`);
  console.log(`   - Price (TL): ${product.price.toFixed(2)}`);
  console.log(`   - Buy Price (TL): ${product.buyPrice.toFixed(2)}`);
  console.log(`   - Sell Price (TL): ${product.sellPrice.toFixed(2)}`);
  console.log(`   - Change 24h: ${product.change24h}%`);
  console.log(`   - Currency: ${product.currency}`);
});

console.log('\n✅ Test Complete!');
