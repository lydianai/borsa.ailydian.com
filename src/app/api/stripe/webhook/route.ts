/**
 * Stripe Webhook Handler
 *
 * White-hat compliance: Handles legitimate subscription lifecycle events
 * Processes: subscription created, updated, deleted, trial_will_end, invoice paid/failed
 */

import { NextRequest, NextResponse } from 'next/server';
import { headers } from 'next/headers';
import Stripe from 'stripe';
import { getStripe } from '@/lib/stripe/config';

export async function POST(req: NextRequest) {
  const body = await req.text();
  const headersList = await headers();
  const signature = headersList.get('stripe-signature');

  if (!signature) {
    return NextResponse.json(
      { error: 'Missing stripe-signature header' },
      { status: 400 }
    );
  }

  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) {
    console.error('STRIPE_WEBHOOK_SECRET not configured');
    return NextResponse.json(
      { error: 'Webhook secret not configured' },
      { status: 500 }
    );
  }

  let event: Stripe.Event;

  try {
    const stripe = getStripe();
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return NextResponse.json(
      { error: 'Invalid signature' },
      { status: 400 }
    );
  }

  try {
    switch (event.type) {
      case 'customer.subscription.created':
        await handleSubscriptionCreated(event.data.object as Stripe.Subscription);
        break;

      case 'customer.subscription.updated':
        await handleSubscriptionUpdated(event.data.object as Stripe.Subscription);
        break;

      case 'customer.subscription.deleted':
        await handleSubscriptionDeleted(event.data.object as Stripe.Subscription);
        break;

      case 'customer.subscription.trial_will_end':
        await handleTrialWillEnd(event.data.object as Stripe.Subscription);
        break;

      case 'invoice.paid':
        await handleInvoicePaid(event.data.object as Stripe.Invoice);
        break;

      case 'invoice.payment_failed':
        await handleInvoicePaymentFailed(event.data.object as Stripe.Invoice);
        break;

      case 'checkout.session.completed':
        await handleCheckoutSessionCompleted(event.data.object as Stripe.Checkout.Session);
        break;

      default:
        console.log(`Unhandled event type: ${event.type}`);
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error('Webhook handler error:', error);
    return NextResponse.json(
      { error: 'Webhook handler failed' },
      { status: 500 }
    );
  }
}

async function handleSubscriptionCreated(subscription: Stripe.Subscription) {
  console.log('Subscription created:', subscription.id);

  // TODO: Update user in database
  // const userId = subscription.metadata.userId;
  // await prisma.subscription.create({
  //   data: {
  //     userId,
  //     stripeSubscriptionId: subscription.id,
  //     stripeCustomerId: subscription.customer as string,
  //     stripePriceId: subscription.items.data[0].price.id,
  //     status: subscription.status,
  //     plan: subscription.metadata.tier || 'starter',
  //     currentPeriodStart: new Date(subscription.current_period_start * 1000),
  //     currentPeriodEnd: new Date(subscription.current_period_end * 1000),
  //     amount: subscription.items.data[0].price.unit_amount || 0,
  //     currency: subscription.currency,
  //   },
  // });

  console.log('✅ Subscription created in database (placeholder)');
}

async function handleSubscriptionUpdated(subscription: Stripe.Subscription) {
  console.log('Subscription updated:', subscription.id);

  // TODO: Update subscription in database
  // await prisma.subscription.update({
  //   where: { stripeSubscriptionId: subscription.id },
  //   data: {
  //     status: subscription.status,
  //     currentPeriodStart: new Date(subscription.current_period_start * 1000),
  //     currentPeriodEnd: new Date(subscription.current_period_end * 1000),
  //     cancelAtPeriodEnd: subscription.cancel_at_period_end,
  //   },
  // });

  console.log('✅ Subscription updated in database (placeholder)');
}

async function handleSubscriptionDeleted(subscription: Stripe.Subscription) {
  console.log('Subscription deleted:', subscription.id);

  // TODO: Update subscription status to canceled
  // await prisma.subscription.update({
  //   where: { stripeSubscriptionId: subscription.id },
  //   data: { status: 'canceled' },
  // });

  // Downgrade user to free tier
  // await prisma.user.update({
  //   where: { id: subscription.metadata.userId },
  //   data: {
  //     subscriptionTier: 'free',
  //     subscriptionStatus: 'inactive',
  //   },
  // });

  console.log('✅ Subscription canceled, user downgraded to free (placeholder)');
}

async function handleTrialWillEnd(subscription: Stripe.Subscription) {
  console.log('Trial will end soon:', subscription.id);

  // TODO: Send email notification to user
  // const user = await prisma.user.findFirst({
  //   where: { stripeCustomerId: subscription.customer as string },
  // });

  // if (user?.email) {
  //   await sendEmail({
  //     to: user.email,
  //     subject: 'Your trial ends soon',
  //     template: 'trial-ending',
  //     data: {
  //       trialEndDate: new Date(subscription.trial_end! * 1000),
  //     },
  //   });
  // }

  console.log('✅ Trial ending notification sent (placeholder)');
}

async function handleInvoicePaid(invoice: Stripe.Invoice) {
  console.log('Invoice paid:', invoice.id);

  // TODO: Update subscription status to active
  // if (invoice.subscription) {
  //   await prisma.subscription.update({
  //     where: { stripeSubscriptionId: invoice.subscription as string },
  //     data: { status: 'active' },
  //   });
  // }

  console.log('✅ Invoice paid, subscription activated (placeholder)');
}

async function handleInvoicePaymentFailed(invoice: Stripe.Invoice) {
  console.log('Invoice payment failed:', invoice.id);

  // TODO: Update subscription status and notify user
  // if (invoice.subscription) {
  //   await prisma.subscription.update({
  //     where: { stripeSubscriptionId: invoice.subscription as string },
  //     data: { status: 'past_due' },
  //   });
  // }

  // Send payment failed notification
  // const user = await prisma.user.findFirst({
  //   where: { stripeCustomerId: invoice.customer as string },
  // });

  // if (user?.email) {
  //   await sendEmail({
  //     to: user.email,
  //     subject: 'Payment failed',
  //     template: 'payment-failed',
  //   });
  // }

  console.log('⚠️ Payment failed, user notified (placeholder)');
}

async function handleCheckoutSessionCompleted(session: Stripe.Checkout.Session) {
  console.log('Checkout session completed:', session.id);

  // TODO: Create or update user subscription
  // const userId = session.metadata?.userId;
  // const tier = session.metadata?.tier;

  // if (userId && tier) {
  //   await prisma.user.update({
  //     where: { id: userId },
  //     data: {
  //       stripeCustomerId: session.customer as string,
  //       subscriptionTier: tier,
  //       subscriptionStatus: 'trialing', // 14-day trial
  //       trialEndsAt: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000),
  //     },
  //   });
  // }

  console.log('✅ Checkout completed, user upgraded (placeholder)');
}
