/**
 * Stripe Checkout API
 *
 * White-hat compliance: Legitimate SaaS subscription checkout
 * Creates Stripe checkout session for subscription billing
 */

import { NextRequest, NextResponse } from 'next/server';
import { getStripe, SUBSCRIPTION_TIERS, type SubscriptionTier } from '@/lib/stripe/config';

export async function POST(req: NextRequest) {
  try {
    const { tier, billingCycle } = await req.json();

    // Validate tier
    if (!tier || !(tier in SUBSCRIPTION_TIERS)) {
      return NextResponse.json(
        { error: 'Invalid subscription tier' },
        { status: 400 }
      );
    }

    // Free tier doesn't need checkout
    if (tier === 'free') {
      return NextResponse.json(
        { error: 'Free tier does not require checkout' },
        { status: 400 }
      );
    }

    const stripe = getStripe();
    const tierConfig = SUBSCRIPTION_TIERS[tier as SubscriptionTier];

    // Validate price ID exists
    if (!tierConfig.priceId) {
      return NextResponse.json(
        { error: 'Stripe price ID not configured for this tier' },
        { status: 500 }
      );
    }

    // Get user info from session (TODO: implement proper auth)
    // For now, we'll use a placeholder email
    const userEmail = 'user@example.com'; // TODO: Get from authenticated session

    // Create checkout session
    const session = await stripe.checkout.sessions.create({
      customer_email: userEmail,
      line_items: [
        {
          price: tierConfig.priceId,
          quantity: 1,
        },
      ],
      mode: 'subscription',
      success_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/dashboard?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/pricing`,
      subscription_data: {
        trial_period_days: 14, // 14-day free trial
        metadata: {
          tier,
          billingCycle,
        },
      },
      metadata: {
        tier,
        billingCycle,
      },
      allow_promotion_codes: true,
      billing_address_collection: 'required',
      payment_method_types: ['card'],
    });

    return NextResponse.json({ url: session.url });
  } catch (error) {
    console.error('Stripe checkout error:', error);

    return NextResponse.json(
      { error: 'Failed to create checkout session' },
      { status: 500 }
    );
  }
}
