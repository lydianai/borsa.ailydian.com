/**
 * NextAuth.js v5 Beta - Auth Instance
 *
 * In next-auth v5, we use NextAuth() to create an auth instance
 * and export the auth() function for server-side session handling
 */

import NextAuth from 'next-auth';
import { authConfig } from './config';

export const { handlers, signIn, signOut, auth } = NextAuth(authConfig);
