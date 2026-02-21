/**
 * Hybrid AI Decision Engine
 * Combines TensorFlow LSTM + Transformer + Decision Forest (Random Forest)
 * For more robust trading signals with ensemble voting
 */

// TensorFlow removed for Vercel deployment

interface MarketData {
  symbol: string;
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface DecisionTreeNode {
  feature: number;
  threshold: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
  value?: number[]; // [BUY, SELL, HOLD] probabilities
}

interface RandomForestConfig {
  numTrees: number;
  maxDepth: number;
  minSamplesSplit: number;
  maxFeatures: number;
}

interface HybridSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  modelVotes: {
    lstm: { action: string; score: number };
    transformer: { action: string; score: number };
    randomForest: { action: string; score: number };
    consensus: number; // 0-1, how much models agree
  };
  targetPrice: number;
  stopLoss: number;
  timeframe: '1d' | '4h' | '1h' | '15m';
  reasoning: string[];
  indicators: any;
}

export class HybridDecisionEngine {
  private randomForest: DecisionTreeNode[] = [];
  private forestConfig: RandomForestConfig = {
    numTrees: 100,
    maxDepth: 15,
    minSamplesSplit: 5,
    maxFeatures: 5,
  };
  private isForestTrained = false;

  constructor(config?: Partial<RandomForestConfig>) {
    if (config) {
      this.forestConfig = { ...this.forestConfig, ...config };
    }
  }

  /**
   * Build Decision Tree recursively
   */
  private buildDecisionTree(
    features: number[][],
    labels: number[][],
    depth: number = 0
  ): DecisionTreeNode {
    // Stop conditions
    if (
      depth >= this.forestConfig.maxDepth ||
      features.length <= this.forestConfig.minSamplesSplit ||
      this.isPure(labels)
    ) {
      return this.createLeafNode(labels);
    }

    // Find best split
    const bestSplit = this.findBestSplit(features, labels);

    if (!bestSplit) {
      return this.createLeafNode(labels);
    }

    // Split data
    const { leftFeatures, leftLabels, rightFeatures, rightLabels } = this.splitData(
      features,
      labels,
      bestSplit.feature,
      bestSplit.threshold
    );

    // Recursively build left and right subtrees
    const leftChild = this.buildDecisionTree(leftFeatures, leftLabels, depth + 1);
    const rightChild = this.buildDecisionTree(rightFeatures, rightLabels, depth + 1);

    return {
      feature: bestSplit.feature,
      threshold: bestSplit.threshold,
      left: leftChild,
      right: rightChild,
    };
  }

  /**
   * Check if labels are pure (all same class)
   */
  private isPure(labels: number[][]): boolean {
    if (labels.length === 0) return true;

    const firstClass = labels[0].indexOf(Math.max(...labels[0]));
    return labels.every(label => label.indexOf(Math.max(...label)) === firstClass);
  }

  /**
   * Create leaf node with majority vote
   */
  private createLeafNode(labels: number[][]): DecisionTreeNode {
    const classCounts = [0, 0, 0]; // BUY, SELL, HOLD

    for (const label of labels) {
      const maxIndex = label.indexOf(Math.max(...label));
      classCounts[maxIndex]++;
    }

    const total = labels.length || 1;
    const probabilities = classCounts.map(count => count / total);

    return { feature: -1, threshold: 0, value: probabilities };
  }

  /**
   * Find best feature and threshold for splitting
   */
  private findBestSplit(
    features: number[][],
    labels: number[][]
  ): { feature: number; threshold: number; gini: number } | null {
    let bestGini = Infinity;
    let bestFeature = -1;
    let bestThreshold = 0;

    // Randomly select features (Random Forest characteristic)
    const numFeatures = features[0]?.length || 0;
    const selectedFeatures = this.randomSelectFeatures(
      numFeatures,
      this.forestConfig.maxFeatures
    );

    for (const featureIdx of selectedFeatures) {
      const values = features.map(f => f[featureIdx]);
      const uniqueValues = Array.from(new Set(values)).sort((a, b) => a - b);

      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;

        const { leftLabels, rightLabels } = this.splitData(
          features,
          labels,
          featureIdx,
          threshold
        );

        if (leftLabels.length === 0 || rightLabels.length === 0) continue;

        const gini = this.calculateGiniImpurity(leftLabels, rightLabels);

        if (gini < bestGini) {
          bestGini = gini;
          bestFeature = featureIdx;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeature === -1) return null;

    return { feature: bestFeature, threshold: bestThreshold, gini: bestGini };
  }

  /**
   * Randomly select features for splitting
   */
  private randomSelectFeatures(total: number, maxSelect: number): number[] {
    const count = Math.min(maxSelect, total);
    const indices = Array.from({ length: total }, (_, i) => i);

    // Fisher-Yates shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    return indices.slice(0, count);
  }

  /**
   * Split data based on feature and threshold
   */
  private splitData(
    features: number[][],
    labels: number[][],
    featureIdx: number,
    threshold: number
  ): {
    leftFeatures: number[][];
    leftLabels: number[][];
    rightFeatures: number[][];
    rightLabels: number[][];
  } {
    const leftFeatures: number[][] = [];
    const leftLabels: number[][] = [];
    const rightFeatures: number[][] = [];
    const rightLabels: number[][] = [];

    for (let i = 0; i < features.length; i++) {
      if (features[i][featureIdx] <= threshold) {
        leftFeatures.push(features[i]);
        leftLabels.push(labels[i]);
      } else {
        rightFeatures.push(features[i]);
        rightLabels.push(labels[i]);
      }
    }

    return { leftFeatures, leftLabels, rightFeatures, rightLabels };
  }

  /**
   * Calculate Gini impurity for split
   */
  private calculateGiniImpurity(leftLabels: number[][], rightLabels: number[][]): number {
    const total = leftLabels.length + rightLabels.length;
    const leftWeight = leftLabels.length / total;
    const rightWeight = rightLabels.length / total;

    const leftGini = this.giniIndex(leftLabels);
    const rightGini = this.giniIndex(rightLabels);

    return leftWeight * leftGini + rightWeight * rightGini;
  }

  /**
   * Calculate Gini index for labels
   */
  private giniIndex(labels: number[][]): number {
    if (labels.length === 0) return 0;

    const classCounts = [0, 0, 0];
    for (const label of labels) {
      const maxIndex = label.indexOf(Math.max(...label));
      classCounts[maxIndex]++;
    }

    let gini = 1;
    for (const count of classCounts) {
      const p = count / labels.length;
      gini -= p * p;
    }

    return gini;
  }

  /**
   * Predict using single decision tree
   */
  private predictTree(tree: DecisionTreeNode, features: number[]): number[] {
    if (tree.value) {
      return tree.value;
    }

    if (features[tree.feature] <= tree.threshold) {
      return tree.left ? this.predictTree(tree.left, features) : [0, 0, 1]; // Default HOLD
    } else {
      return tree.right ? this.predictTree(tree.right, features) : [0, 0, 1];
    }
  }

  /**
   * Train Random Forest on historical data
   */
  trainRandomForest(trainingData: { features: number[][]; labels: number[][] }) {
    console.log('🌲 Training Random Forest with', this.forestConfig.numTrees, 'trees...');
    this.randomForest = [];

    for (let i = 0; i < this.forestConfig.numTrees; i++) {
      // Bootstrap sampling
      const { sampledFeatures, sampledLabels } = this.bootstrapSample(
        trainingData.features,
        trainingData.labels
      );

      // Build tree
      const tree = this.buildDecisionTree(sampledFeatures, sampledLabels);
      this.randomForest.push(tree);

      if (i % 20 === 0) {
        console.log(`  Tree ${i}/${this.forestConfig.numTrees} trained`);
      }
    }

    this.isForestTrained = true;
    console.log('✅ Random Forest training complete');
  }

  /**
   * Bootstrap sampling (sampling with replacement)
   */
  private bootstrapSample(
    features: number[][],
    labels: number[][]
  ): { sampledFeatures: number[][]; sampledLabels: number[][] } {
    const n = features.length;
    const sampledFeatures: number[][] = [];
    const sampledLabels: number[][] = [];

    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      sampledFeatures.push(features[idx]);
      sampledLabels.push(labels[idx]);
    }

    return { sampledFeatures, sampledLabels };
  }

  /**
   * Predict using Random Forest (ensemble voting)
   */
  predictRandomForest(features: number[]): { action: 'BUY' | 'SELL' | 'HOLD'; score: number } {
    if (!this.isForestTrained || this.randomForest.length === 0) {
      return { action: 'HOLD', score: 0.5 };
    }

    const votes = [0, 0, 0]; // BUY, SELL, HOLD

    for (const tree of this.randomForest) {
      const prediction = this.predictTree(tree, features);
      const maxIndex = prediction.indexOf(Math.max(...prediction));
      votes[maxIndex]++;
    }

    const maxVotes = Math.max(...votes);
    const winnerIndex = votes.indexOf(maxVotes);
    const confidence = maxVotes / this.randomForest.length;

    const actions: Array<'BUY' | 'SELL' | 'HOLD'> = ['BUY', 'SELL', 'HOLD'];

    return {
      action: actions[winnerIndex],
      score: confidence,
    };
  }

  /**
   * Get ensemble consensus from all models
   */
  calculateConsensus(
    lstmAction: string,
    transformerAction: string,
    forestAction: string
  ): number {
    const actions = [lstmAction, transformerAction, forestAction];
    const uniqueActions = new Set(actions);

    // All agree = 1.0
    if (uniqueActions.size === 1) return 1.0;

    // 2 out of 3 agree = 0.67
    if (uniqueActions.size === 2) return 0.67;

    // All disagree = 0.33
    return 0.33;
  }

  /**
   * Generate hybrid signal with ensemble voting
   */
  generateHybridSignal(
    lstmPrediction: { action: string; score: number },
    transformerPrediction: { action: string; score: number },
    features: number[],
    indicators: any,
    currentPrice: number
  ): HybridSignal {
    // Random Forest prediction
    const forestPrediction = this.predictRandomForest(features);

    // Calculate consensus
    const consensus = this.calculateConsensus(
      lstmPrediction.action,
      transformerPrediction.action,
      forestPrediction.action
    );

    // Weighted voting (higher weight for models with high confidence)
    const lstmWeight = 0.4 * lstmPrediction.score;
    const transformerWeight = 0.3 * transformerPrediction.score;
    const forestWeight = 0.3 * forestPrediction.score;

    const totalWeight = lstmWeight + transformerWeight + forestWeight;

    const buyScore =
      (lstmPrediction.action === 'BUY' ? lstmWeight : 0) +
      (transformerPrediction.action === 'BUY' ? transformerWeight : 0) +
      (forestPrediction.action === 'BUY' ? forestWeight : 0);

    const sellScore =
      (lstmPrediction.action === 'SELL' ? lstmWeight : 0) +
      (transformerPrediction.action === 'SELL' ? transformerWeight : 0) +
      (forestPrediction.action === 'SELL' ? forestWeight : 0);

    const holdScore =
      (lstmPrediction.action === 'HOLD' ? lstmWeight : 0) +
      (transformerPrediction.action === 'HOLD' ? transformerWeight : 0) +
      (forestPrediction.action === 'HOLD' ? forestWeight : 0);

    const maxScore = Math.max(buyScore, sellScore, holdScore);
    const finalAction =
      maxScore === buyScore ? 'BUY' : maxScore === sellScore ? 'SELL' : 'HOLD';

    const finalConfidence = maxScore / totalWeight;

    // Generate reasoning
    const reasoning: string[] = [];
    reasoning.push(`🤖 LSTM Model: ${lstmPrediction.action} (${(lstmPrediction.score * 100).toFixed(1)}%)`);
    reasoning.push(`🔮 Transformer: ${transformerPrediction.action} (${(transformerPrediction.score * 100).toFixed(1)}%)`);
    reasoning.push(`🌲 Random Forest: ${forestPrediction.action} (${(forestPrediction.score * 100).toFixed(1)}%)`);
    reasoning.push(`🎯 Model Consensus: ${(consensus * 100).toFixed(0)}%`);

    if (consensus >= 0.67) {
      reasoning.push('✅ Strong model agreement - High reliability');
    } else if (consensus >= 0.5) {
      reasoning.push('⚠️ Moderate agreement - Use caution');
    } else {
      reasoning.push('❌ Low agreement - High uncertainty');
    }

    // Calculate ATR for stop loss/target
    const atr = indicators.atr || currentPrice * 0.02;
    const targetPrice =
      finalAction === 'BUY'
        ? currentPrice + atr * 2.5
        : finalAction === 'SELL'
        ? currentPrice - atr * 2.5
        : currentPrice;

    const stopLoss =
      finalAction === 'BUY'
        ? currentPrice - atr * 1.5
        : finalAction === 'SELL'
        ? currentPrice + atr * 1.5
        : currentPrice;

    return {
      symbol: '',
      action: finalAction,
      confidence: finalConfidence,
      modelVotes: {
        lstm: { action: lstmPrediction.action, score: lstmPrediction.score },
        transformer: { action: transformerPrediction.action, score: transformerPrediction.score },
        randomForest: { action: forestPrediction.action, score: forestPrediction.score },
        consensus,
      },
      targetPrice,
      stopLoss,
      timeframe: '1h',
      reasoning,
      indicators,
    };
  }
}

// Singleton instance
let hybridEngineInstance: HybridDecisionEngine | null = null;

export function getHybridEngine(config?: Partial<RandomForestConfig>): HybridDecisionEngine {
  if (!hybridEngineInstance) {
    hybridEngineInstance = new HybridDecisionEngine(config);
  }
  return hybridEngineInstance;
}