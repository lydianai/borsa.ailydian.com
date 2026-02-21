/**
 * NextAuth Type Extensions
 *
 * White-hat compliance: Type-safe authentication
 */

import 'next-auth';

declare module 'next-auth' {
  interface Session {
    user: {
      id: string;
      email: string;
      name?: string | null;
      username?: string;
      role?: string;
      isAdmin?: boolean;
      isApproved?: boolean;
      hasActivePayment?: boolean;
      subscriptionTier?: string;
    };
  }

  interface User {
    id: string;
    email: string;
    name?: string | null;
    username?: string;
    role?: string;
    isAdmin?: boolean;
    isApproved?: boolean;
    hasActivePayment?: boolean;
    subscriptionTier?: string;
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    id?: string;
    email?: string;
    username?: string;
    role?: string;
    isAdmin?: boolean;
    isApproved?: boolean;
    hasActivePayment?: boolean;
    subscriptionTier?: string;
  }
}
