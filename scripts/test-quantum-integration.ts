#!/usr/bin/env tsx

/**
 * BlueQubit Quantum Integration Test Script
 * SarDag Emrah platformu için kuantum entegrasyon testi
 */

import { blueQubitClient } from "../src/lib/quantum/bluequbit-client.js";
import { config } from "dotenv";

// Load environment variables
config({ path: ".env.bluequbit" });

async function testQuantumIntegration(): Promise<boolean> {
  console.log("🚀 BlueQubit Quantum Integration Test");
  console.log("=====================================\n");

  // 1. Configuration Check
  console.log("📋 Configuration Check:");
  const clientConfig = blueQubitClient.getConfig();
  console.log(`  API URL: ${clientConfig.apiUrl}`);
  console.log(`  Web URL: ${clientConfig.webUrl}`);
  console.log(`  Simulator Qubits: ${clientConfig.simulatorQubits}`);
  console.log(`  GPU Simulator Qubits: ${clientConfig.gpuSimulatorQubits}`);
  console.log(`  Hardware Type: ${clientConfig.hardwareType}`);
  console.log(`  Max Qubits: ${clientConfig.maxQubits}`);
  console.log(`  Environment: ${clientConfig.environment}`);
  console.log(`  API Key: ${clientConfig.apiKey ? "✅ Set" : "❌ Missing"}\n`);

  // 2. API Key Check
  if (
    !clientConfig.apiKey ||
    clientConfig.apiKey === "your_bluequbit_api_key_here"
  ) {
    console.log("❌ CRITICAL: BlueQubit API key is missing!");
    console.log("Please get your API key from https://app.bluequbit.io");
    console.log("And add it to .env.bluequbit file:");
    console.log("BLUEQUBIT_API_KEY=your_actual_api_key_here\n");
    return false;
  }

  // 3. Initialization Test
  console.log("🔧 Initializing BlueQubit Client...");
  try {
    const initialized = await blueQubitClient.initialize();
    if (initialized) {
      console.log("✅ BlueQubit client initialized successfully!\n");
    } else {
      console.log("❌ Failed to initialize BlueQubit client\n");
      return false;
    }
  } catch (error) {
    console.log(`❌ Initialization error: ${(error as Error).message}\n`);
    return false;
  }

  // 4. Backend List Test
  console.log("🖥️  Getting Available Quantum Backends...");
  try {
    const backends = await blueQubitClient.getAvailableBackends();
    console.log(`✅ Found ${backends.length} quantum backends:`);
    backends.forEach((backend, index) => {
      console.log(`  ${index + 1}. ${backend.name} (${backend.type})`);
      console.log(`     Qubits: ${backend.qubits}`);
      console.log(`     Available: ${backend.available ? "✅" : "❌"}`);
      if (backend.fidelity) console.log(`     Fidelity: ${backend.fidelity}`);
      if (backend.costPerHour)
        console.log(`     Cost: $${backend.costPerHour}/hour`);
      if (backend.queueSize)
        console.log(`     Queue: ${backend.queueSize} jobs`);
    });
    console.log("");
  } catch (error) {
    console.log(`❌ Failed to get backends: ${(error as Error).message}\n`);
  }

  // 5. Budget Check
  console.log("💰 Checking Budget Status...");
  try {
    const budgetOk = await blueQubitClient.checkBudget();
    console.log(
      `Budget Status: ${budgetOk ? "✅ Within limits" : "⚠️ Exceeded limits"}\n`,
    );
  } catch (error) {
    console.log(`❌ Budget check failed: ${(error as Error).message}\n`);
  }

  // 6. Usage Stats
  console.log("📊 Getting Usage Statistics...");
  try {
    const stats = await blueQubitClient.getUsageStats();
    if (stats) {
      console.log("✅ Usage Statistics:");
      console.log(`  Hourly Cost: $${stats.hourlyCost || 0}`);
      console.log(`  Jobs Run: ${stats.totalJobs || 0}`);
      console.log(`  Quantum Circuits: ${stats.totalCircuits || 0}`);
      console.log(`  API Calls: ${stats.apiCalls || 0}`);
    } else {
      console.log("ℹ️  No usage statistics available");
    }
    console.log("");
  } catch (error) {
    console.log(`❌ Failed to get usage stats: ${(error as Error).message}\n`);
  }

  // 7. Ready Status
  console.log("🎯 Final Status Check:");
  const isReady = blueQubitClient.isReady();
  console.log(`BlueQubit Client Ready: ${isReady ? "✅" : "❌"}\n`);

  if (isReady) {
    console.log("🎉 BlueQubit Quantum Integration is READY!");
    console.log("Next steps:");
    console.log("1. Test portfolio optimization");
    console.log("2. Test risk analysis");
    console.log("3. Integrate with trading strategies");
    console.log("4. Create quantum dashboard UI");
  } else {
    console.log("⚠️  BlueQubit Quantum Integration needs attention");
  }

  return isReady;
}

// Portfolio Optimization Test
async function testPortfolioOptimization(): Promise<boolean> {
  console.log("📈 Testing Portfolio Optimization...");

  const sampleRequest = {
    assets: ["BTC", "ETH", "ADA", "DOT", "LINK"],
    expectedReturns: [0.15, 0.25, 0.35, 0.45, 0.55],
    covarianceMatrix: [
      [0.04, 0.02, 0.01, 0.03, 0.02],
      [0.02, 0.09, 0.03, 0.04, 0.02],
      [0.01, 0.03, 0.16, 0.05, 0.03],
      [0.03, 0.04, 0.05, 0.25, 0.04],
      [0.02, 0.02, 0.03, 0.04, 0.36],
    ],
    riskTolerance: 0.1,
    constraints: {
      minWeight: 0.05,
      maxWeight: 0.4,
      targetReturn: 0.3,
    },
  };

  try {
    const result = await blueQubitClient.optimizePortfolio(sampleRequest);
    console.log("✅ Portfolio Optimization Results:");
    console.log(
      `  Optimal Weights: [${result.optimalWeights.map((w) => (w * 100).toFixed(2) + "%").join(", ")}]`,
    );
    console.log(
      `  Expected Return: ${(result.expectedReturn * 100).toFixed(2)}%`,
    );
    console.log(`  Risk: ${(result.risk * 100).toFixed(2)}%`);
    console.log(`  Sharpe Ratio: ${result.sharpeRatio.toFixed(4)}`);
    console.log(`  Quantum Advantage: ${result.quantumAdvantage}x`);
    console.log(`  Execution Time: ${result.executionTime}ms`);
    return true;
  } catch (error) {
    console.log(
      `❌ Portfolio optimization failed: ${(error as Error).message}`,
    );
    return false;
  }
}

// Risk Analysis Test
async function testRiskAnalysis(): Promise<boolean> {
  console.log("⚡ Testing Risk Analysis...");

  const sampleRequest = {
    portfolio: {
      assets: ["BTC", "ETH", "ADA"],
      weights: [0.5, 0.3, 0.2],
    },
    confidenceLevel: 0.95,
    timeHorizon: 30, // days
    scenarios: 10000,
  };

  try {
    const result = await blueQubitClient.analyzeRisk(sampleRequest);
    console.log("✅ Risk Analysis Results:");
    console.log(
      `  Value at Risk (95%): ${(result.valueAtRisk * 100).toFixed(2)}%`,
    );
    console.log(
      `  Conditional VaR: ${(result.conditionalVaR * 100).toFixed(2)}%`,
    );
    console.log(
      `  Expected Shortfall: ${(result.expectedShortfall * 100).toFixed(2)}%`,
    );
    console.log(`  Quantum Speedup: ${result.quantumSpeedup}x`);
    console.log(
      `  Confidence Interval: [${(result.confidenceIntervals.lower * 100).toFixed(2)}%, ${(result.confidenceIntervals.upper * 100).toFixed(2)}%]`,
    );
    return true;
  } catch (error) {
    console.log(`❌ Risk analysis failed: ${(error as Error).message}`);
    return false;
  }
}

// Main execution
async function main(): Promise<void> {
  const basicTest = await testQuantumIntegration();

  if (basicTest) {
    console.log("\n🧪 Running Advanced Tests...\n");

    const portfolioTest = await testPortfolioOptimization();
    console.log("");

    const riskTest = await testRiskAnalysis();
    console.log("");

    if (portfolioTest && riskTest) {
      console.log(
        "🎉 ALL TESTS PASSED! Quantum integration is fully functional!",
      );
    } else {
      console.log(
        "⚠️  Some advanced tests failed. Check API limits and configuration.",
      );
    }
  }
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { testQuantumIntegration, testPortfolioOptimization, testRiskAnalysis };
