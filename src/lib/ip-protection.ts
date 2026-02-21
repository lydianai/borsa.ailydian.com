/**
 * IP PROTECTION & OBFUSCATION LAYER
 * White-hat security: Hide proprietary technical implementation details from end-users
 * This prevents reverse engineering while maintaining full functionality
 */

// ============================================
// AI MODEL NAME OBFUSCATION
// ============================================

/**
 * Maps real AI model names to generic aliases
 * Prevents competitors from identifying specific models used
 */
const AI_MODEL_ALIASES: Record<string, string> = {
  // LLM Providers
  'Groq': 'Strategy Engine A',
  'OpenAI': 'Strategy Engine B',
  'Claude': 'Strategy Engine C',
  'GPT': 'Analysis Engine',
  'GPT-4': 'Analysis Engine Pro',
  'GPT-3.5': 'Analysis Engine Standard',

  // Model-specific variants
  'llama': 'Predictive Model A',
  'mistral': 'Predictive Model B',
  'gemini': 'Predictive Model C',
};

/**
 * Maps technical indicator names to generic names
 * Hides the specific technical analysis methods used
 */
const INDICATOR_ALIASES: Record<string, string> = {
  // Moving Averages
  'MACD': 'Trend Indicator',
  'EMA': 'Moving Average',
  'SMA': 'Simple Average',

  // Volatility Indicators
  'Bollinger': 'Volatility Band',
  'Bollinger Bands': 'Volatility Bands',
  'ATR': 'Volatility Measure',
  'Keltner': 'Channel Indicator',

  // Momentum Indicators
  'Stochastic': 'Oscillator A',
  'RSI': 'Oscillator B',
  'CCI': 'Oscillator C',
  'MFI': 'Volume Oscillator',

  // Trend Indicators
  'ADX': 'Trend Strength',
  'Parabolic SAR': 'Trend Tracker',
  'Ichimoku': 'Cloud Indicator',

  // Volume Indicators
  'OBV': 'Volume Trend',
  'VWAP': 'Volume Weighted Price',
  'Accumulation/Distribution': 'Volume Flow',
};

/**
 * Maps Python service names to generic names
 */
const SERVICE_ALIASES: Record<string, string> = {
  'TA-Lib': 'Technical Analysis Service',
  'Signal Generator': 'Strategy Generator',
  'Feature Engineering': 'Data Processor',
  'AI Models': 'Prediction Engine',
  'Risk Management': 'Risk Calculator',
};

// ============================================
// OBFUSCATION FUNCTIONS
// ============================================

/**
 * Replace AI model names with generic aliases
 * Case-insensitive replacement
 */
export function obfuscateAIModel(text: string): string {
  let obfuscated = text;

  for (const [real, alias] of Object.entries(AI_MODEL_ALIASES)) {
    const regex = new RegExp(`\\b${real}\\b`, 'gi');
    obfuscated = obfuscated.replace(regex, alias);
  }

  return obfuscated;
}

/**
 * Replace technical indicator names with generic aliases
 */
export function obfuscateIndicator(text: string): string {
  let obfuscated = text;

  for (const [real, alias] of Object.entries(INDICATOR_ALIASES)) {
    const regex = new RegExp(`\\b${real}\\b`, 'gi');
    obfuscated = obfuscated.replace(regex, alias);
  }

  return obfuscated;
}

/**
 * Replace Python service names with generic aliases
 */
export function obfuscateService(text: string): string {
  let obfuscated = text;

  for (const [real, alias] of Object.entries(SERVICE_ALIASES)) {
    const regex = new RegExp(`\\b${real}\\b`, 'gi');
    obfuscated = obfuscated.replace(regex, alias);
  }

  return obfuscated;
}

/**
 * Comprehensive obfuscation - applies all obfuscation layers
 */
export function obfuscateAll(text: string): string {
  let result = text;
  result = obfuscateAIModel(result);
  result = obfuscateIndicator(result);
  result = obfuscateService(result);
  return result;
}

/**
 * Recursively obfuscate all string values in an object
 * Preserves object structure while sanitizing content
 */
export function sanitizeObject<T extends Record<string, any>>(obj: T): T {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(item => sanitizeObject(item)) as unknown as T;
  }

  const sanitized: Record<string, any> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      // Obfuscate string values
      sanitized[key] = obfuscateAll(value);
    } else if (typeof value === 'object' && value !== null) {
      // Recursively sanitize nested objects
      sanitized[key] = sanitizeObject(value);
    } else {
      // Preserve non-string primitives
      sanitized[key] = value;
    }
  }

  return sanitized as T;
}

// ============================================
// API RESPONSE SANITIZATION
// ============================================

/**
 * Sanitize API response before sending to client
 * White-hat approach: Remove technical details while preserving functionality
 *
 * Usage:
 * ```typescript
 * return NextResponse.json(sanitizeAPIResponse({
 *   success: true,
 *   data: { strategies: [...] }
 * }))
 * ```
 */
export function sanitizeAPIResponse<T extends { success: boolean; data?: any; error?: string }>(
  response: T
): T {
  if (!response.data) {
    return response;
  }

  return {
    ...response,
    data: sanitizeObject(response.data),
    error: response.error ? obfuscateAll(response.error) : undefined,
  };
}

/**
 * Remove specific sensitive fields from strategy objects
 * Used for strategy analysis responses
 */
export function sanitizeStrategy(strategy: any): any {
  const {
    // Remove internal implementation details
    indicatorValues,
    rawIndicators,
    technicalDetails,
    modelConfig,
    internalScore,
    ...publicFields
  } = strategy;

  // Obfuscate remaining text fields
  return sanitizeObject(publicFields);
}

/**
 * Sanitize strategy reason to hide indicator names
 * Example: "RSI oversold (34.5)" → "Momentum Indicator oversold (34.5)"
 */
export function sanitizeStrategyReason(reason: string): string {
  let sanitized = obfuscateAll(reason);

  // Remove specific numerical thresholds that reveal strategy logic
  // Keep general direction but hide exact parameters
  sanitized = sanitized.replace(/\b\d+\.\d+\s*(threshold|level|value)\b/gi, 'threshold');

  return sanitized;
}

// ============================================
// EXPORT UTILITY FUNCTIONS
// ============================================

/**
 * Check if a string contains any sensitive technical terms
 */
export function containsSensitiveTerms(text: string): boolean {
  const allTerms = [
    ...Object.keys(AI_MODEL_ALIASES),
    ...Object.keys(INDICATOR_ALIASES),
    ...Object.keys(SERVICE_ALIASES),
  ];

  const lowerText = text.toLowerCase();
  return allTerms.some(term => lowerText.includes(term.toLowerCase()));
}

/**
 * Get list of all protected terms (for debugging/testing)
 */
export function getProtectedTerms(): {
  models: string[];
  indicators: string[];
  services: string[];
} {
  return {
    models: Object.keys(AI_MODEL_ALIASES),
    indicators: Object.keys(INDICATOR_ALIASES),
    services: Object.keys(SERVICE_ALIASES),
  };
}
