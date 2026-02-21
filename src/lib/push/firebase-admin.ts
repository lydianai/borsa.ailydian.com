/**
 * 🔥 FIREBASE ADMIN INITIALIZATION
 * Server-side Firebase Admin SDK for FCM push notifications
 *
 * Features:
 * - Singleton pattern (one instance per server)
 * - Lazy initialization
 * - Service account authentication
 * - White-hat compliance: All initialization logged
 */

import * as admin from 'firebase-admin';

// ============================================================================
// FIREBASE ADMIN SINGLETON
// ============================================================================

let firebaseApp: admin.app.App | null = null;

/**
 * Initialize Firebase Admin SDK
 * Uses service account JSON from environment variable
 */
export function initializeFirebase(): admin.app.App {
  // Return existing instance if already initialized
  if (firebaseApp) {
    return firebaseApp;
  }

  try {
    const serviceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT;

    if (!serviceAccountJson) {
      console.warn('[Firebase] FIREBASE_SERVICE_ACCOUNT not set, push notifications disabled');
      throw new Error('Firebase service account not configured');
    }

    // Parse service account JSON
    const serviceAccount = JSON.parse(serviceAccountJson);

    // Initialize Firebase Admin
    firebaseApp = admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
      projectId: serviceAccount.project_id,
    });

    console.log('[Firebase] ✅ Initialized successfully');
    console.log(`[Firebase] Project ID: ${serviceAccount.project_id}`);

    return firebaseApp;
  } catch (error: any) {
    console.error('[Firebase] Initialization failed:', error.message);
    throw error;
  }
}

/**
 * Get Firebase Admin instance (lazy initialization)
 */
export function getFirebaseAdmin(): admin.app.App {
  if (!firebaseApp) {
    return initializeFirebase();
  }

  return firebaseApp;
}

/**
 * Get Firebase Messaging instance
 */
export function getMessaging(): admin.messaging.Messaging {
  const app = getFirebaseAdmin();
  return admin.messaging(app);
}

/**
 * Check if Firebase is initialized and available
 */
export function isFirebaseAvailable(): boolean {
  try {
    return !!process.env.FIREBASE_SERVICE_ACCOUNT && firebaseApp !== null;
  } catch {
    return false;
  }
}

/**
 * Shutdown Firebase Admin (for testing or graceful shutdown)
 */
export async function shutdownFirebase(): Promise<void> {
  if (firebaseApp) {
    await firebaseApp.delete();
    firebaseApp = null;
    console.log('[Firebase] Shutdown complete');
  }
}
