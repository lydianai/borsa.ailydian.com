#!/usr/bin/env python3
"""Test signal generation manually"""

import requests
import sys

try:
    print("🔄 Testing signal generation for BTC...")

    # Step 1: Get price data
    print("\n1. Fetching price data...")
    price_response = requests.get(
        "http://localhost:3000/api/trading/top100",
        params={'limit': 100},
        timeout=10
    )

    print(f"   Status: {price_response.status_code}")
    print(f"   OK: {price_response.ok}")

    if not price_response.ok:
        print(f"   ❌ Failed to get price")
        sys.exit(1)

    price_data = price_response.json()
    print(f"   Success: {price_data.get('success')}")
    print(f"   Coins: {len(price_data.get('data', []))}")

    # Step 2: Find BTC
    print("\n2. Finding BTC...")
    coin_info = None
    for item in price_data.get('data', []):
        if item['coin']['symbol'] == 'BTC':
            coin_info = item['coin']
            break

    if not coin_info:
        print("   ❌ BTC not found")
        sys.exit(1)

    print(f"   ✅ Found: {coin_info['name']}")
    print(f"   Price: ${coin_info['price']:,.2f}")

    # Step 3: Get AI prediction
    print("\n3. Getting AI prediction...")
    pred_response = requests.post(
        "http://localhost:5003/predict/single",
        json={
            'symbol': 'BTC',
            'timeframe': '1h',
            'model': 'lstm_standard'
        },
        timeout=30
    )

    print(f"   Status: {pred_response.status_code}")

    if not pred_response.ok:
        print(f"   ❌ Failed to get prediction")
        sys.exit(1)

    pred_data = pred_response.json()
    print(f"   Success: {pred_data.get('success')}")
    print(f"   Action: {pred_data['prediction']['action']}")
    print(f"   Confidence: {pred_data['prediction']['confidence']:.4f}")

    print("\n✅ ALL TESTS PASSED!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
