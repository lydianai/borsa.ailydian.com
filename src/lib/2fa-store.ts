/**
 * 🔐 2FA STORAGE SYSTEM
 * Google Authenticator için güvenli secret yönetimi
 */

import { promises as fs } from 'fs';
import path from 'path';

export interface User2FAData {
  username: string;
  secret: string;
  enabled: boolean;
  backupCodes: string[];
  createdAt: string;
  lastVerified?: string;
}

const STORE_FILE = path.join(process.cwd(), 'data', '2fa-secrets.json');

// In-memory cache for performance
let cache: Map<string, User2FAData> = new Map();
let cacheLoaded = false;

/**
 * Ensure data directory exists
 */
async function ensureDataDir() {
  const dataDir = path.join(process.cwd(), 'data');
  try {
    await fs.mkdir(dataDir, { recursive: true });
  } catch (error) {
    // Directory already exists
  }
}

/**
 * Load 2FA data from file
 */
async function loadStore(): Promise<Map<string, User2FAData>> {
  if (cacheLoaded && cache.size > 0) {
    return cache;
  }

  await ensureDataDir();

  try {
    const data = await fs.readFile(STORE_FILE, 'utf-8');
    const parsed = JSON.parse(data);
    cache = new Map(Object.entries(parsed));
    cacheLoaded = true;
  } catch (error) {
    // File doesn't exist yet, start with empty cache
    cache = new Map();
    cacheLoaded = true;
  }

  return cache;
}

/**
 * Save 2FA data to file
 */
async function saveStore() {
  await ensureDataDir();
  const obj = Object.fromEntries(cache);
  await fs.writeFile(STORE_FILE, JSON.stringify(obj, null, 2), 'utf-8');
}

/**
 * Get user's 2FA data
 */
export async function get2FAData(username: string): Promise<User2FAData | null> {
  const store = await loadStore();
  return store.get(username) || null;
}

/**
 * Save user's 2FA data
 */
export async function save2FAData(username: string, data: User2FAData): Promise<void> {
  const store = await loadStore();
  store.set(username, data);
  cache = store;
  await saveStore();
}

/**
 * Check if user has 2FA enabled
 */
export async function is2FAEnabled(username: string): Promise<boolean> {
  const data = await get2FAData(username);
  return data?.enabled || false;
}

/**
 * Disable 2FA for user
 */
export async function disable2FA(username: string): Promise<void> {
  const store = await loadStore();
  store.delete(username);
  cache = store;
  await saveStore();
}

/**
 * Generate backup codes (8 codes, 8 characters each)
 */
export function generateBackupCodes(): string[] {
  const codes: string[] = [];
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // No confusing chars

  for (let i = 0; i < 8; i++) {
    let code = '';
    for (let j = 0; j < 8; j++) {
      code += chars[Math.floor(Math.random() * chars.length)];
    }
    // Format: XXXX-XXXX
    codes.push(`${code.slice(0, 4)}-${code.slice(4)}`);
  }

  return codes;
}

/**
 * Verify and consume a backup code
 */
export async function verifyBackupCode(username: string, code: string): Promise<boolean> {
  const data = await get2FAData(username);
  if (!data) return false;

  const normalizedCode = code.toUpperCase().replace(/-/g, '');
  const codeIndex = data.backupCodes.findIndex(
    (bc) => bc.replace(/-/g, '') === normalizedCode
  );

  if (codeIndex === -1) return false;

  // Remove used backup code
  data.backupCodes.splice(codeIndex, 1);
  data.lastVerified = new Date().toISOString();
  await save2FAData(username, data);

  return true;
}
