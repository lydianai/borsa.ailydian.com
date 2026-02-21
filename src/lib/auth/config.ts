/**
 * NextAuth.js Configuration
 *
 * White-hat compliance: Secure authentication for legitimate SaaS platform
 * Features: Email/password, email verification, admin approval, payment verification
 */

import { NextAuthConfig } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import { PrismaAdapter } from '@auth/prisma-adapter';
import { prisma } from '@/lib/prisma';
import bcrypt from 'bcryptjs';

export const authConfig: NextAuthConfig = {
  adapter: PrismaAdapter(prisma),
  providers: [
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error('Email ve şifre gereklidir');
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email as string },
        });

        if (!user) {
          throw new Error('Kullanıcı bulunamadı');
        }

        // Check email verification
        if (!user.emailVerified) {
          throw new Error('Email adresiniz doğrulanmamış. Lütfen emailinizi kontrol edin.');
        }

        // Check admin approval
        if (!user.isApproved && !user.isAdmin) {
          throw new Error('Hesabınız henüz onaylanmamış. Admin onayı bekleniyor.');
        }

        // Verify password
        const isPasswordValid = await bcrypt.compare(
          credentials.password as string,
          user.passwordHash
        );

        if (!isPasswordValid) {
          throw new Error('Geçersiz şifre');
        }

        // Update last login
        await prisma.user.update({
          where: { id: user.id },
          data: { lastLoginAt: new Date() },
        });

        return {
          id: user.id,
          email: user.email,
          name: user.username,
          role: user.role,
          isAdmin: user.isAdmin,
          isApproved: user.isApproved,
          hasActivePayment: user.hasActivePayment,
          subscriptionTier: user.subscriptionTier,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.role = user.role;
        token.isAdmin = user.isAdmin;
        token.isApproved = user.isApproved;
        token.hasActivePayment = user.hasActivePayment;
        token.subscriptionTier = user.subscriptionTier;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
        session.user.role = token.role as string;
        session.user.isAdmin = token.isAdmin as boolean;
        session.user.isApproved = token.isApproved as boolean;
        session.user.hasActivePayment = token.hasActivePayment as boolean;
        session.user.subscriptionTier = token.subscriptionTier as string;
      }
      return session;
    },
  },
  pages: {
    signIn: '/login',
    error: '/login',
  },
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },
  secret: process.env.NEXTAUTH_SECRET,
};
