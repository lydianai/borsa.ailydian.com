/**
 * Reinforcement Learning Trading Agent
 * Based on TF-Agents: Deep Q-Network (DQN) for trading decisions
 * Learns optimal buy/sell/hold strategy through environment interaction
 */

// TensorFlow removed for Vercel deployment

interface State {
  prices: number[]; // Last N prices
  indicators: {
    rsi: number;
    macd: number;
    volume: number;
    volatility: number;
  };
  position: 'none' | 'long'; // Current position
  balance: number;
  holdings: number;
}

interface Action {
  type: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
}

interface Experience {
  state: State;
  action: Action;
  reward: number;
  nextState: State;
  done: boolean;
}

interface RLConfig {
  stateSize: number;
  actionSize: number;
  learningRate: number;
  gamma: number; // Discount factor
  epsilon: number; // Exploration rate
  epsilonDecay: number;
  epsilonMin: number;
  memorySize: number;
  batchSize: number;
}

export class ReinforcementLearningAgent {
  private config: RLConfig;
  private qNetwork: any | null = null;
  private targetNetwork: any | null = null;
  private memory: Experience[] = [];
  private epsilon: number;
  private episodeCount = 0;
  private totalReward = 0;

  constructor(config?: Partial<RLConfig>) {
    this.config = {
      stateSize: 20, // prices + indicators
      actionSize: 3, // BUY, SELL, HOLD
      learningRate: 0.001,
      gamma: 0.95,
      epsilon: 1.0,
      epsilonDecay: 0.995,
      epsilonMin: 0.01,
      memorySize: 10000,
      batchSize: 32,
      ...config,
    };

    this.epsilon = this.config.epsilon;
  }

  /**
   * Build Deep Q-Network
   * Q(s, a) = expected cumulative reward for action a in state s
   */
  async buildModel(): Promise<void> {
    // TensorFlow removed for Vercel deployment
    this.qNetwork = null;
    this.targetNetwork = null;
    console.warn('⚠️ Reinforcement Learning Agent disabled - TensorFlow removed for Vercel deployment');
  }

  /**
   * Clone model for target network
   */
  private async cloneModel(_model: any): Promise<any> {
    // TensorFlow removed for Vercel deployment
    return null;
  }

  /**
   * Encode state to feature vector
   */
  private encodeState(state: State): number[] {
    const features: number[] = [];

    // Price features (normalized)
    const prices = state.prices;
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    for (const price of prices.slice(-10)) {
      features.push((price - minPrice) / (maxPrice - minPrice + 1e-8));
    }

    // Indicator features (normalized)
    features.push(state.indicators.rsi / 100);
    features.push((state.indicators.macd + 100) / 200); // Normalize MACD
    features.push(Math.min(state.indicators.volume / 1000000, 1)); // Cap volume
    features.push(Math.min(state.indicators.volatility, 1));

    // Position encoding
    features.push(state.position === 'long' ? 1 : 0);

    // Portfolio features
    features.push(state.balance / 10000); // Normalize balance
    features.push(state.holdings / 10); // Normalize holdings

    // Pad to state size
    while (features.length < this.config.stateSize) {
      features.push(0);
    }

    return features.slice(0, this.config.stateSize);
  }

  /**
   * Select action using epsilon-greedy policy
   */
  async selectAction(_state: State): Promise<Action> {
    // TensorFlow removed for Vercel deployment
    // Return random action
    const randomAction = Math.floor(Math.random() * this.config.actionSize);
    const actions: Array<'BUY' | 'SELL' | 'HOLD'> = ['BUY', 'SELL', 'HOLD'];

    return {
      type: actions[randomAction],
      confidence: 0.3,
    };
  }

  /**
   * Calculate reward based on action outcome
   */
  calculateReward(
    action: Action,
    prevState: State,
    newState: State,
    marketChange: number
  ): number {
    let reward = 0;

    if (action.type === 'BUY' && prevState.position === 'none') {
      // Bought: reward based on price movement
      reward = marketChange * 100; // Scale reward
    } else if (action.type === 'SELL' && prevState.position === 'long') {
      // Sold: reward based on profit
      const profit = (newState.balance - prevState.balance) / prevState.balance;
      reward = profit * 1000; // Scale reward
    } else if (action.type === 'HOLD') {
      // Hold: small penalty for inaction if big move happened
      reward = Math.abs(marketChange) > 0.02 ? -1 : 0;
    } else {
      // Invalid action (e.g., buy when already long)
      reward = -10;
    }

    // Penalize large losses
    if (newState.balance < prevState.balance * 0.9) {
      reward -= 50;
    }

    // Reward large gains
    if (newState.balance > prevState.balance * 1.1) {
      reward += 50;
    }

    return reward;
  }

  /**
   * Store experience in replay memory
   */
  remember(experience: Experience): void {
    this.memory.push(experience);

    // Keep memory size within limit
    if (this.memory.length > this.config.memorySize) {
      this.memory.shift();
    }
  }

  /**
   * Train Q-Network using experience replay
   */
  async replay(): Promise<void> {
    // TensorFlow removed for Vercel deployment
    // Stub implementation
    if (this.epsilon > this.config.epsilonMin) {
      this.epsilon *= this.config.epsilonDecay;
    }
  }

  /**
   * Sample random batch from memory
   */
  private sampleBatch(batchSize: number): Experience[] {
    const batch: Experience[] = [];
    const indices = new Set<number>();

    while (indices.size < batchSize && indices.size < this.memory.length) {
      const idx = Math.floor(Math.random() * this.memory.length);
      if (!indices.has(idx)) {
        indices.add(idx);
        batch.push(this.memory[idx]);
      }
    }

    return batch;
  }

  /**
   * Update target network (soft update)
   */
  async updateTargetNetwork(_tau: number = 0.01): Promise<void> {
    // TensorFlow removed for Vercel deployment
    // Stub implementation
  }

  /**
   * Train agent on historical episodes
   */
  async train(episodes: number = 1000): Promise<void> {
    console.log(`🎓 Training RL Agent for ${episodes} episodes...`);

    for (let episode = 0; episode < episodes; episode++) {
      // Simulate episode here (in production, use real market data)
      // For now, just perform replay on existing memory

      await this.replay();

      if (episode % 10 === 0) {
        await this.updateTargetNetwork();
      }

      if (episode % 100 === 0) {
        console.log(
          `  Episode ${episode}/${episodes} | ` +
          `Epsilon: ${this.epsilon.toFixed(3)} | ` +
          `Memory: ${this.memory.length}`
        );
      }

      this.episodeCount++;
    }

    console.log('✅ Training complete');
  }

  /**
   * Get agent statistics
   */
  getStats(): {
    episodeCount: number;
    totalReward: number;
    epsilon: number;
    memorySize: number;
    averageReward: number;
  } {
    return {
      episodeCount: this.episodeCount,
      totalReward: this.totalReward,
      epsilon: this.epsilon,
      memorySize: this.memory.length,
      averageReward: this.episodeCount > 0 ? this.totalReward / this.episodeCount : 0,
    };
  }

  /**
   * Save agent models
   */
  async save(path: string): Promise<void> {
    if (!this.qNetwork) {
      throw new Error('Q-Network not initialized');
    }

    await this.qNetwork.save(`file://${path}/q-network`);
    console.log(`✅ RL Agent saved to ${path}`);
  }

  /**
   * Load agent models
   */
  async load(_path: string): Promise<void> {
    // TensorFlow removed for Vercel deployment
    throw new Error('RL Agent loading disabled - TensorFlow removed for Vercel deployment');
  }

  /**
   * Dispose agent and cleanup
   */
  dispose(): void {
    if (this.qNetwork) {
      this.qNetwork.dispose();
      this.qNetwork = null;
    }
    if (this.targetNetwork) {
      this.targetNetwork.dispose();
      this.targetNetwork = null;
    }
    this.memory = [];
  }
}

// Singleton instance
let agentInstance: ReinforcementLearningAgent | null = null;

export function getRLAgent(config?: Partial<RLConfig>): ReinforcementLearningAgent {
  if (!agentInstance) {
    agentInstance = new ReinforcementLearningAgent(config);
  }
  return agentInstance;
}