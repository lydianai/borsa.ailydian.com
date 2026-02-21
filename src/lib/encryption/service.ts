/**
 * Encryption Service for Exchange API Keys
 *
 * White-hat compliance: Secure storage of user-provided exchange API credentials
 *
 * Security features:
 * - AES-256-GCM encryption
 * - Random IV per encryption
 * - Environment-based master key
 * - Never logs decrypted values
 *
 * CRITICAL: ENCRYPTION_KEY must be 32 bytes (256 bits)
 * Generate with: openssl rand -base64 32
 */

import crypto from 'crypto';

const ALGORITHM = 'aes-256-gcm';
const IV_LENGTH = 16; // 128 bits
const _AUTH_TAG_LENGTH = 16; // 128 bits
const _SALT_LENGTH = 32; // 256 bits

/**
 * Get master encryption key from environment
 * CRITICAL: Must be set in production
 */
function getMasterKey(): Buffer {
  const key = process.env.ENCRYPTION_KEY;

  if (!key) {
    throw new Error(
      'ENCRYPTION_KEY not set in environment. ' +
      'Generate with: openssl rand -base64 32'
    );
  }

  // Derive a 32-byte key from the env variable
  return crypto.pbkdf2Sync(
    key,
    'ailydian-exchange-api-salt', // Application-specific salt
    100000, // iterations
    32, // key length in bytes
    'sha256'
  );
}

/**
 * Encrypt sensitive data using AES-256-GCM
 *
 * @param plaintext - The data to encrypt (API key, secret, passphrase)
 * @returns Object containing encrypted data, IV, and auth tag
 */
export function encrypt(plaintext: string): {
  encryptedData: string;
  iv: string;
  authTag: string;
} {
  try {
    // Validate input
    if (!plaintext || plaintext.trim() === '') {
      throw new Error('Cannot encrypt empty string');
    }

    const masterKey = getMasterKey();
    const iv = crypto.randomBytes(IV_LENGTH);

    // Create cipher
    const cipher = crypto.createCipheriv(ALGORITHM, masterKey, iv);

    // Encrypt
    let encrypted = cipher.update(plaintext, 'utf8', 'base64');
    encrypted += cipher.final('base64');

    // Get authentication tag
    const authTag = cipher.getAuthTag();

    return {
      encryptedData: encrypted,
      iv: iv.toString('base64'),
      authTag: authTag.toString('base64'),
    };
  } catch (error) {
    console.error('Encryption error (no sensitive data logged)');
    throw new Error('Encryption failed');
  }
}

/**
 * Decrypt sensitive data using AES-256-GCM
 *
 * @param encryptedData - Base64 encoded encrypted data
 * @param iv - Base64 encoded initialization vector
 * @param authTag - Base64 encoded authentication tag
 * @returns Decrypted plaintext
 */
export function decrypt(
  encryptedData: string,
  iv: string,
  authTag: string
): string {
  try {
    // Validate inputs
    if (!encryptedData || !iv || !authTag) {
      throw new Error('Missing required decryption parameters');
    }

    const masterKey = getMasterKey();
    const ivBuffer = Buffer.from(iv, 'base64');
    const authTagBuffer = Buffer.from(authTag, 'base64');

    // Create decipher
    const decipher = crypto.createDecipheriv(ALGORITHM, masterKey, ivBuffer);
    decipher.setAuthTag(authTagBuffer);

    // Decrypt
    let decrypted = decipher.update(encryptedData, 'base64', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  } catch (error) {
    console.error('Decryption error (no sensitive data logged)');
    throw new Error('Decryption failed - data may be corrupted');
  }
}

/**
 * Encrypt exchange API credentials for database storage
 *
 * @param apiKey - Exchange API key
 * @param apiSecret - Exchange API secret
 * @param passphrase - Optional passphrase (for OKX, etc.)
 * @returns Encrypted credentials ready for database
 */
export function encryptExchangeCredentials(
  apiKey: string,
  apiSecret: string,
  passphrase?: string
): {
  encryptedApiKey: string;
  encryptedApiSecret: string;
  encryptedPassphrase: string | null;
  encryptionIV: string;
} {
  // Use a single IV for all three fields for simplicity
  // This is secure as each field is independently encrypted
  const iv = crypto.randomBytes(IV_LENGTH).toString('base64');

  const keyResult = encrypt(apiKey);
  const secretResult = encrypt(apiSecret);
  const passphraseResult = passphrase ? encrypt(passphrase) : null;

  // Combine encrypted data with auth tag (format: encryptedData:authTag)
  return {
    encryptedApiKey: `${keyResult.encryptedData}:${keyResult.authTag}`,
    encryptedApiSecret: `${secretResult.encryptedData}:${secretResult.authTag}`,
    encryptedPassphrase: passphraseResult
      ? `${passphraseResult.encryptedData}:${passphraseResult.authTag}`
      : null,
    encryptionIV: iv,
  };
}

/**
 * Decrypt exchange API credentials from database
 *
 * @param encryptedApiKey - Encrypted API key from DB
 * @param encryptedApiSecret - Encrypted API secret from DB
 * @param encryptedPassphrase - Encrypted passphrase from DB (optional)
 * @param iv - Initialization vector from DB
 * @returns Decrypted credentials (NEVER log these!)
 */
export function decryptExchangeCredentials(
  encryptedApiKey: string,
  encryptedApiSecret: string,
  iv: string,
  encryptedPassphrase?: string | null
): {
  apiKey: string;
  apiSecret: string;
  passphrase: string | null;
} {
  try {
    // Split encrypted data and auth tag
    const [keyData, keyAuthTag] = encryptedApiKey.split(':');
    const [secretData, secretAuthTag] = encryptedApiSecret.split(':');

    if (!keyData || !keyAuthTag || !secretData || !secretAuthTag) {
      throw new Error('Invalid encrypted credential format');
    }

    const apiKey = decrypt(keyData, iv, keyAuthTag);
    const apiSecret = decrypt(secretData, iv, secretAuthTag);

    let passphrase: string | null = null;
    if (encryptedPassphrase) {
      const [passphraseData, passphraseAuthTag] = encryptedPassphrase.split(':');
      if (passphraseData && passphraseAuthTag) {
        passphrase = decrypt(passphraseData, iv, passphraseAuthTag);
      }
    }

    return { apiKey, apiSecret, passphrase };
  } catch (error) {
    console.error('Failed to decrypt exchange credentials');
    throw new Error('Unable to decrypt credentials - please re-enter your API keys');
  }
}

/**
 * Validate encryption configuration
 * Call this on startup to ensure encryption is properly configured
 */
export function validateEncryptionConfig(): boolean {
  try {
    const testData = 'test-encryption-key-validation';
    const encrypted = encrypt(testData);
    const decrypted = decrypt(
      encrypted.encryptedData,
      encrypted.iv,
      encrypted.authTag
    );

    if (decrypted !== testData) {
      throw new Error('Encryption validation failed - decrypt mismatch');
    }

    return true;
  } catch (error) {
    console.error('Encryption configuration validation failed:', error);
    return false;
  }
}

/**
 * Securely compare two strings in constant time
 * Prevents timing attacks when comparing API keys
 */
export function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }

  return crypto.timingSafeEqual(
    Buffer.from(a, 'utf8'),
    Buffer.from(b, 'utf8')
  );
}
