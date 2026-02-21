/**
 * NextAuth.js v5 API Route Handler
 *
 * White-hat compliance: Handles authentication endpoints
 * Updated for next-auth v5 beta - uses handlers from auth.ts
 */

import { handlers } from '@/lib/auth/auth';

export const { GET, POST } = handlers;
