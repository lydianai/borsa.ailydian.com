/**
 * QUANTUM TRADING BOT - LSTM Neural Network
 * Time series prediction for cryptocurrency trading
 */

import * as tf from '@tensorflow/tfjs-node';

interface Prediction {
  symbol: string;
  prediction: number; // 0-1 scale (0=strong sell, 0.5=neutral, 1=strong buy)
  confidence: number;
  timestamp: number;
}

export class QuantumTradingBot {
  private model: tf.LayersModel | null = null;
  private readonly SEQUENCE_LENGTH = 60; // 60 candles for prediction

  constructor() {
    this.initializeModel();
  }

  private async initializeModel() {
    try {
      // LSTM Model for time series prediction
      this.model = tf.sequential({
        layers: [
          // LSTM Layer 1
          tf.layers.lstm({
            units: 128,
            returnSequences: true,
            inputShape: [this.SEQUENCE_LENGTH, 5] // OHLCV
          }),
          tf.layers.dropout({ rate: 0.2 }),

          // LSTM Layer 2
          tf.layers.lstm({
            units: 64,
            returnSequences: false
          }),
          tf.layers.dropout({ rate: 0.2 }),

          // Dense layers
          tf.layers.dense({ units: 32, activation: 'relu' }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({ units: 16, activation: 'relu' }),
          tf.layers.dense({ units: 1, activation: 'sigmoid' }) // 0-1 prediction
        ]
      });

      this.model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
      });

      console.log('✅ Quantum Trading Bot (LSTM) initialized');
    } catch (error) {
      console.error('❌ LSTM initialization failed:', error);
    }
  }

  /**
   * Predict for single symbol
   */
  async predict(symbol: string, ohlcv: number[][]): Promise<Prediction> {
    if (!this.model) {
      return {
        symbol,
        prediction: 0.5,
        confidence: 0,
        timestamp: Date.now()
      };
    }

    try {
      // Normalize OHLCV data
      const normalized = this.normalizeData(ohlcv);

      // Prepare input tensor [1, SEQUENCE_LENGTH, 5]
      const input = tf.tensor3d([normalized], [1, this.SEQUENCE_LENGTH, 5]);

      // Get prediction
      const predictionTensor = this.model.predict(input) as tf.Tensor;
      const predictionValue = (await predictionTensor.data())[0];

      // Calculate confidence based on distance from 0.5
      const confidence = Math.abs(predictionValue - 0.5) * 2;

      input.dispose();
      predictionTensor.dispose();

      return {
        symbol,
        prediction: predictionValue,
        confidence,
        timestamp: Date.now()
      };
    } catch (error) {
      console.error(`Error predicting ${symbol}:`, error);
      return {
        symbol,
        prediction: 0.5,
        confidence: 0,
        timestamp: Date.now()
      };
    }
  }

  /**
   * Predict for all top symbols (mock for now)
   */
  async predictAll(): Promise<Prediction[]> {
    // In production, this would fetch real OHLCV data
    // For now, return mock predictions
    const symbols = ['BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI'];

    return symbols.map(symbol => {
      // Generate realistic-looking predictions
      const randomPrediction = 0.3 + Math.random() * 0.4; // 0.3-0.7 range
      const confidence = Math.abs(randomPrediction - 0.5) * 2;

      return {
        symbol,
        prediction: randomPrediction,
        confidence,
        timestamp: Date.now()
      };
    });
  }

  /**
   * Normalize OHLCV data to 0-1 range
   */
  private normalizeData(ohlcv: number[][]): number[][] {
    if (ohlcv.length === 0) {
      return Array(this.SEQUENCE_LENGTH).fill([0, 0, 0, 0, 0]);
    }

    // Get last SEQUENCE_LENGTH candles
    const recent = ohlcv.slice(-this.SEQUENCE_LENGTH);

    // Find min/max for normalization
    const allPrices = recent.flatMap(candle => [candle[1], candle[2], candle[3], candle[4]]);
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const priceRange = maxPrice - minPrice || 1;

    const allVolumes = recent.map(candle => candle[5]);
    const maxVolume = Math.max(...allVolumes) || 1;

    // Normalize each candle
    return recent.map(candle => {
      return [
        (candle[1] - minPrice) / priceRange, // Open
        (candle[2] - minPrice) / priceRange, // High
        (candle[3] - minPrice) / priceRange, // Low
        (candle[4] - minPrice) / priceRange, // Close
        candle[5] / maxVolume                 // Volume
      ];
    });
  }

  /**
   * Train model with historical data
   */
  async train(trainingData: { ohlcv: number[][], label: number }[]) {
    if (!this.model) {
      console.error('Model not initialized');
      return;
    }

    console.log('🎓 Training LSTM model...');

    // Prepare training tensors
    const xs = trainingData.map(d => this.normalizeData(d.ohlcv));
    const ys = trainingData.map(d => d.label);

    const xTensor = tf.tensor3d(xs, [xs.length, this.SEQUENCE_LENGTH, 5]);
    const yTensor = tf.tensor2d(ys, [ys.length, 1]);

    // Train
    await this.model.fit(xTensor, yTensor, {
      epochs: 50,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 10 === 0) {
            console.log(`Epoch ${epoch}: loss=${logs?.loss.toFixed(4)}, acc=${logs?.acc.toFixed(4)}`);
          }
        }
      }
    });

    xTensor.dispose();
    yTensor.dispose();

    console.log('✅ LSTM training complete');
  }
}

// Singleton instance
export const quantumBot = new QuantumTradingBot();
