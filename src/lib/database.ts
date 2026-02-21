/**
 * DATABASE LAYER
 * File-based persistent storage with encryption
 * Vercel-compatible, works in both local and production
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';

const DATA_DIR = path.join(process.cwd(), '.data');
const ENCRYPTION_KEY = process.env.DATABASE_ENCRYPTION_KEY || 'default-key-change-in-production';

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

interface DatabaseStore {
  [key: string]: any;
}

/**
 * Encrypt data before saving
 */
function encrypt(data: string): string {
  const iv = crypto.randomBytes(16);
  const key = crypto.createHash('sha256').update(ENCRYPTION_KEY).digest();
  const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
  let encrypted = cipher.update(data, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  return iv.toString('hex') + ':' + encrypted;
}

/**
 * Decrypt data after loading
 */
function decrypt(data: string): string {
  const parts = data.split(':');
  const iv = Buffer.from(parts[0], 'hex');
  const encryptedData = parts[1];
  const key = crypto.createHash('sha256').update(ENCRYPTION_KEY).digest();
  const decipher = crypto.createDecipheriv('aes-256-cbc', key, iv);
  let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  return decrypted;
}

/**
 * Database class for persistent storage
 */
export class Database {
  private filePath: string;
  private data: DatabaseStore = {};

  constructor(collectionName: string) {
    this.filePath = path.join(DATA_DIR, `${collectionName}.enc`);
    this.load();
  }

  /**
   * Load data from file
   */
  private load(): void {
    try {
      if (fs.existsSync(this.filePath)) {
        const encryptedData = fs.readFileSync(this.filePath, 'utf8');
        const decryptedData = decrypt(encryptedData);
        this.data = JSON.parse(decryptedData);
      }
    } catch (error) {
      console.error(`[Database] Error loading ${this.filePath}:`, error);
      this.data = {};
    }
  }

  /**
   * Save data to file
   */
  private save(): void {
    try {
      const jsonData = JSON.stringify(this.data, null, 2);
      const encryptedData = encrypt(jsonData);
      fs.writeFileSync(this.filePath, encryptedData, 'utf8');
    } catch (error) {
      console.error(`[Database] Error saving ${this.filePath}:`, error);
    }
  }

  /**
   * Get value by key
   */
  get(key: string): any {
    return this.data[key];
  }

  /**
   * Set value by key
   */
  set(key: string, value: any): void {
    this.data[key] = value;
    this.save();
  }

  /**
   * Delete value by key
   */
  delete(key: string): void {
    delete this.data[key];
    this.save();
  }

  /**
   * Check if key exists
   */
  has(key: string): boolean {
    return key in this.data;
  }

  /**
   * Get all keys
   */
  keys(): string[] {
    return Object.keys(this.data);
  }

  /**
   * Get all values
   */
  values(): any[] {
    return Object.values(this.data);
  }

  /**
   * Get all entries
   */
  entries(): [string, any][] {
    return Object.entries(this.data);
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.data = {};
    this.save();
  }

  /**
   * Get size
   */
  size(): number {
    return Object.keys(this.data).length;
  }
}

/**
 * Database instances for each collection
 */
export const riskManagementDB = new Database('risk-management');
export const notificationChannelsDB = new Database('notification-channels');
export const strategyManagementDB = new Database('strategy-management');
export const apiKeysDB = new Database('api-keys');
export const watchlistFiltersDB = new Database('watchlist-filters');
export const performanceAnalyticsDB = new Database('performance-analytics');
export const automationRulesDB = new Database('automation-rules');
export const securitySettingsDB = new Database('security-settings');

/**
 * Migration helper: Convert Map to Database
 */
export function migrateMapToDatabase<T>(
  map: Map<string, T>,
  db: Database
): void {
  map.forEach((value, key) => {
    db.set(key, value);
  });
  console.log(`[Database] Migrated ${map.size} entries to ${db.constructor.name}`);
}

/**
 * Backup all databases
 */
export function backupAllDatabases(): string {
  const backup: Record<string, any> = {
    'risk-management': riskManagementDB.entries(),
    'notification-channels': notificationChannelsDB.entries(),
    'strategy-management': strategyManagementDB.entries(),
    'api-keys': apiKeysDB.entries(),
    'watchlist-filters': watchlistFiltersDB.entries(),
    'performance-analytics': performanceAnalyticsDB.entries(),
    'automation-rules': automationRulesDB.entries(),
    'security-settings': securitySettingsDB.entries(),
  };

  const backupPath = path.join(DATA_DIR, `backup-${Date.now()}.json`);
  fs.writeFileSync(backupPath, JSON.stringify(backup, null, 2), 'utf8');

  return backupPath;
}

/**
 * Restore from backup
 */
export function restoreFromBackup(backupPath: string): void {
  try {
    const backup = JSON.parse(fs.readFileSync(backupPath, 'utf8'));

    Object.entries(backup).forEach(([collectionName, entries]) => {
      const db = new Database(collectionName);
      db.clear();
      (entries as [string, any][]).forEach(([key, value]) => {
        db.set(key, value);
      });
    });

    console.log('[Database] Restored from backup successfully');
  } catch (error) {
    console.error('[Database] Error restoring from backup:', error);
  }
}

/**
 * Get database stats
 */
export function getDatabaseStats() {
  return {
    'risk-management': riskManagementDB.size(),
    'notification-channels': notificationChannelsDB.size(),
    'strategy-management': strategyManagementDB.size(),
    'api-keys': apiKeysDB.size(),
    'watchlist-filters': watchlistFiltersDB.size(),
    'performance-analytics': performanceAnalyticsDB.size(),
    'automation-rules': automationRulesDB.size(),
    'security-settings': securitySettingsDB.size(),
    totalRecords:
      riskManagementDB.size() +
      notificationChannelsDB.size() +
      strategyManagementDB.size() +
      apiKeysDB.size() +
      watchlistFiltersDB.size() +
      performanceAnalyticsDB.size() +
      automationRulesDB.size() +
      securitySettingsDB.size(),
  };
}
