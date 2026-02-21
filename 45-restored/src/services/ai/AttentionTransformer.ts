/**
 * Advanced Attention Transformer for Market Prediction
 * Based on "Attention Is All You Need" (Vaswani et al. 2017)
 * And recent ArXiv research on financial time-series transformers
 *
 * Multi-Head Self-Attention with Positional Encoding
 */

// TensorFlow removed for Vercel deployment

interface AttentionConfig {
  dModel: number; // Model dimension
  numHeads: number; // Number of attention heads
  dff: number; // Feed-forward network dimension
  numLayers: number; // Number of transformer blocks
  dropoutRate: number;
  maxSeqLength: number;
}

export class AttentionTransformer {
  private config: AttentionConfig;
  private model: any | null = null;
  private isInitialized = false;

  constructor(config?: Partial<AttentionConfig>) {
    this.config = {
      dModel: 128,
      numHeads: 8,
      dff: 512,
      numLayers: 4,
      dropoutRate: 0.1,
      maxSeqLength: 60,
      ...config,
    };
  }

  /**
   * Positional Encoding (sine/cosine)
   * PE(pos, 2i) = sin(pos / 10000^(2i/dModel))
   * PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
   */
  private createPositionalEncoding(seqLength: number, dModel: number): // tf.Tensor2D {
    const positions = // tf.range(0, seqLength, 1, 'float32').expandDims(1);
    const depths = // tf.range(0, dModel, 2, 'float32').div(dModel);

    const angleRates = // tf.pow(10000, depths).reciprocal();
    const angleRads = positions.mul(angleRates);

    // Apply sin to even indices (2i)
    const sines = angleRads.sin();

    // Apply cos to odd indices (2i+1)
    const cosines = angleRads.cos();

    // Interleave sines and cosines
    const posEncoding = // tf.stack([sines, cosines], 2).reshape([seqLength, dModel]);

    return posEncoding as // tf.Tensor2D;
  }

  /**
   * Scaled Dot-Product Attention
   * Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
   */
  private scaledDotProductAttention(
    q: // tf.Tensor3D,
    k: // tf.Tensor3D,
    v: // tf.Tensor3D,
    mask?: // tf.Tensor
  ): { attention: // tf.Tensor3D; weights: // tf.Tensor } {
    // Q * K^T
    const matmulQK = // tf.matMul(q, k, false, true);

    // Scale by sqrt(d_k)
    const dk = // tf.scalar(k.shape[k.shape.length - 1]);
    const scaledAttention = matmulQK.div(// tf.sqrt(dk));

    // Apply mask if provided
    let maskedAttention = scaledAttention;
    if (mask) {
      maskedAttention = scaledAttention.add(mask.mul(-1e9));
    }

    // Softmax normalization
    const attentionWeights = // tf.softmax(maskedAttention, -1);

    // Apply attention to values
    const output = // tf.matMul(attentionWeights, v);

    return {
      attention: output as // tf.Tensor3D,
      weights: attentionWeights,
    };
  }

  /**
   * Multi-Head Attention Layer
   * Allows model to jointly attend to information from different representation subspaces
   */
  private multiHeadAttention(
    inputs: // tf.Tensor3D,
    numHeads: number,
    dModel: number
  ): // tf.Tensor3D {
    const batchSize = inputs.shape[0];
    const seqLength = inputs.shape[1];
    const depth = Math.floor(dModel / numHeads);

    // Linear projections for Q, K, V
    const wq = // tf.layers.dense({ units: dModel, useBias: false }).apply(inputs) as // tf.Tensor3D;
    const wk = // tf.layers.dense({ units: dModel, useBias: false }).apply(inputs) as // tf.Tensor3D;
    const wv = // tf.layers.dense({ units: dModel, useBias: false }).apply(inputs) as // tf.Tensor3D;

    // Split into multiple heads
    const splitHeads = (x: // tf.Tensor3D): // tf.Tensor4D => {
      const reshaped = x.reshape([batchSize, seqLength, numHeads, depth]);
      return reshaped.transpose([0, 2, 1, 3]) as // tf.Tensor4D;
    };

    const q = splitHeads(wq);
    const k = splitHeads(wk);
    const v = splitHeads(wv);

    // Scaled dot-product attention for each head
    const { attention } = this.scaledDotProductAttention(
      q.reshape([batchSize * numHeads, seqLength, depth]) as // tf.Tensor3D,
      k.reshape([batchSize * numHeads, seqLength, depth]) as // tf.Tensor3D,
      v.reshape([batchSize * numHeads, seqLength, depth]) as // tf.Tensor3D
    );

    // Concatenate heads
    const attentionReshaped = attention
      .reshape([batchSize, numHeads, seqLength, depth])
      .transpose([0, 2, 1, 3]);

    const concatAttention = attentionReshaped.reshape([batchSize, seqLength, dModel]);

    // Final linear projection
    const output = // tf.layers.dense({ units: dModel }).apply(concatAttention) as // tf.Tensor3D;

    return output;
  }

  /**
   * Feed-Forward Network
   * FFN(x) = max(0, xW1 + b1)W2 + b2
   */
  private feedForwardNetwork(dModel: number, dff: number): any {
    return // tf.sequential({
      layers: [
        // tf.layers.dense({ units: dff, activation: 'relu' }),
        // tf.layers.dense({ units: dModel }),
      ],
    });
  }

  /**
   * Transformer Encoder Block
   * - Multi-head self-attention
   * - Add & Norm (residual connection + layer normalization)
   * - Feed-forward network
   * - Add & Norm
   */
  private transformerEncoderBlock(
    inputs: // tf.Tensor3D,
    numHeads: number,
    dModel: number,
    dff: number,
    dropoutRate: number
  ): // tf.Tensor3D {
    // Multi-head attention
    const attentionOutput = this.multiHeadAttention(inputs, numHeads, dModel);

    // Dropout
    const attentionDropout = // tf.layers.dropout({ rate: dropoutRate }).apply(attentionOutput) as // tf.Tensor3D;

    // Add & Norm (residual connection)
    const norm1 = // tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(
      inputs.add(attentionDropout)
    ) as // tf.Tensor3D;

    // Feed-forward network
    const ffn = this.feedForwardNetwork(dModel, dff);
    const ffnOutput = ffn.apply(norm1) as // tf.Tensor3D;

    // Dropout
    const ffnDropout = // tf.layers.dropout({ rate: dropoutRate }).apply(ffnOutput) as // tf.Tensor3D;

    // Add & Norm
    const norm2 = // tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(
      norm1.add(ffnDropout)
    ) as // tf.Tensor3D;

    return norm2;
  }

  /**
   * Build complete Transformer model
   */
  async buildModel(inputFeatures: number): Promise<void> {
    const { dModel, numHeads, dff, numLayers, dropoutRate, maxSeqLength } = this.config;

    const input = // tf.input({ shape: [maxSeqLength, inputFeatures] });

    // Input embedding
    let x = // tf.layers.dense({ units: dModel, activation: 'relu' }).apply(input) as // tf.Tensor3D;

    // Add positional encoding
    const posEncoding = this.createPositionalEncoding(maxSeqLength, dModel);
    x = x.add(posEncoding) as // tf.Tensor3D;

    // Stack transformer encoder blocks
    for (let i = 0; i < numLayers; i++) {
      x = this.transformerEncoderBlock(x, numHeads, dModel, dff, dropoutRate);
    }

    // Global average pooling
    const pooled = // tf.layers.globalAveragePooling1d().apply(x) as // tf.Tensor2D;

    // Classification head
    const dense1 = // tf.layers.dense({ units: 64, activation: 'relu' }).apply(pooled);
    const dropout = // tf.layers.dropout({ rate: dropoutRate }).apply(dense1);
    const output = // tf.layers.dense({ units: 3, activation: 'softmax', name: 'output' }).apply(dropout);

    this.model = // tf.model({ inputs: input, outputs: output as // tf.SymbolicTensor });

    this.model.compile({
      optimizer: // tf.train.adam(0.0001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });

    this.isInitialized = true;
    console.log('✅ Advanced Attention Transformer initialized');
    console.log(`   - Model dimension: ${dModel}`);
    console.log(`   - Attention heads: ${numHeads}`);
    console.log(`   - Transformer blocks: ${numLayers}`);
    console.log(`   - Feed-forward dim: ${dff}`);
  }

  /**
   * Train model on historical data
   */
  async train(
    features: number[][][],
    labels: number[][],
    epochs: number = 50,
    batchSize: number = 32
  ): Promise<void> {
    if (!this.model) {
      throw new Error('Model not initialized. Call buildModel() first.');
    }

    const xs = // tf.tensor3d(features);
    const ys = // tf.tensor2d(labels);

    console.log('🎓 Training Attention Transformer...');

    await this.model.fit(xs, ys, {
      epochs,
      batchSize,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 10 === 0) {
            console.log(
              `  Epoch ${epoch}: loss=${logs?.loss.toFixed(4)}, accuracy=${logs?.acc.toFixed(4)}, val_loss=${logs?.val_loss.toFixed(4)}, val_acc=${logs?.val_acc.toFixed(4)}`
            );
          }
        },
      },
    });

    xs.dispose();
    ys.dispose();

    console.log('✅ Training complete');
  }

  /**
   * Predict with attention weights visualization
   */
  async predict(features: number[][]): Promise<{
    prediction: number[];
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
  }> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    const input = // tf.tensor3d([features]);
    const prediction = this.model.predict(input) as // tf.Tensor;
    const predArray = await prediction.data();

    input.dispose();
    prediction.dispose();

    const maxIndex = predArray.indexOf(Math.max(...Array.from(predArray)));
    const actions: Array<'BUY' | 'SELL' | 'HOLD'> = ['BUY', 'SELL', 'HOLD'];

    return {
      prediction: Array.from(predArray),
      action: actions[maxIndex],
      confidence: predArray[maxIndex],
    };
  }

  /**
   * Extract attention weights for interpretability
   * Shows which time steps the model focuses on
   */
  async getAttentionWeights(features: number[][]): Promise<number[][]> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    // Create intermediate model to extract attention weights
    const attentionLayer = this.model.layers.find(l => l.name.includes('attention'));

    if (!attentionLayer) {
      console.warn('No attention layer found for weight extraction');
      return [];
    }

    const intermediateModel = // tf.model({
      inputs: this.model.input,
      outputs: attentionLayer.output as // tf.SymbolicTensor,
    });

    const input = // tf.tensor3d([features]);
    const weights = intermediateModel.predict(input) as // tf.Tensor;
    const weightsArray = await weights.array() as number[][][];

    input.dispose();
    weights.dispose();

    return weightsArray[0] || [];
  }

  /**
   * Save model to disk
   */
  async saveModel(path: string): Promise<void> {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    await this.model.save(`file://${path}`);
    console.log(`✅ Model saved to ${path}`);
  }

  /**
   * Load model from disk
   */
  async loadModel(path: string): Promise<void> {
    this.model = await // tf.loadLayersModel(`file://${path}/model.json`);
    this.isInitialized = true;
    console.log(`✅ Model loaded from ${path}`);
  }

  /**
   * Get model summary
   */
  summary(): void {
    if (!this.model) {
      throw new Error('Model not initialized');
    }

    this.model.summary();
  }

  /**
   * Dispose model and free memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
      this.isInitialized = false;
    }
  }
}

// Singleton instance
let transformerInstance: AttentionTransformer | null = null;

export function getAttentionTransformer(config?: Partial<AttentionConfig>): AttentionTransformer {
  if (!transformerInstance) {
    transformerInstance = new AttentionTransformer(config);
  }
  return transformerInstance;
}

/**
 * Helper: Prepare features for transformer
 * Converts raw OHLCV to normalized feature vectors
 */
export function prepareTransformerFeatures(
  data: Array<{
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>,
  windowSize: number = 60
): number[][] {
  const features: number[][] = [];

  // Take last windowSize candles
  const recentData = data.slice(-windowSize);

  // Normalize
  const closes = recentData.map(d => d.close);
  const volumes = recentData.map(d => d.volume);

  const minPrice = Math.min(...closes);
  const maxPrice = Math.max(...closes);
  const minVol = Math.min(...volumes);
  const maxVol = Math.max(...volumes);

  for (const candle of recentData) {
    const normalizedOpen = (candle.open - minPrice) / (maxPrice - minPrice + 1e-8);
    const normalizedHigh = (candle.high - minPrice) / (maxPrice - minPrice + 1e-8);
    const normalizedLow = (candle.low - minPrice) / (maxPrice - minPrice + 1e-8);
    const normalizedClose = (candle.close - minPrice) / (maxPrice - minPrice + 1e-8);
    const normalizedVolume = (candle.volume - minVol) / (maxVol - minVol + 1e-8);

    features.push([
      normalizedOpen,
      normalizedHigh,
      normalizedLow,
      normalizedClose,
      normalizedVolume,
    ]);
  }

  return features;
}