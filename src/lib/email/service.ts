/**
 * Email Service
 *
 * White-hat compliance: Sends legitimate transactional emails
 * Uses Resend for email delivery
 */

import { Resend } from 'resend';

const resend = new Resend(process.env.RESEND_API_KEY);

const FROM_EMAIL = process.env.EMAIL_FROM || 'noreply@example.com';
const APP_NAME = process.env.NEXT_PUBLIC_APP_NAME || 'LyTrade Scanner';
const APP_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

export interface EmailOptions {
  to: string;
  subject: string;
  html: string;
}

/**
 * Send email using Resend
 */
export async function sendEmail({ to, subject, html }: EmailOptions) {
  try {
    const { data, error } = await resend.emails.send({
      from: FROM_EMAIL,
      to,
      subject,
      html,
    });

    if (error) {
      console.error('Email send error:', error);
      throw new Error(`Email gönderimi başarısız: ${error.message}`);
    }

    console.log('Email sent successfully:', data);
    return data;
  } catch (error) {
    console.error('Email service error:', error);
    throw error;
  }
}

/**
 * Send email verification link
 */
export async function sendVerificationEmail(email: string, token: string) {
  const verifyUrl = `${APP_URL}/verify-email?token=${token}`;

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
          .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
          .button { display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
          .footer { text-align: center; margin-top: 20px; color: #666; font-size: 12px; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🎯 ${APP_NAME}</h1>
          </div>
          <div class="content">
            <h2>Email Adresinizi Doğrulayın</h2>
            <p>Merhaba,</p>
            <p>${APP_NAME} platformuna hoş geldiniz! Hesabınızı aktifleştirmek için email adresinizi doğrulamanız gerekmektedir.</p>
            <p>Aşağıdaki butona tıklayarak email adresinizi doğrulayabilirsiniz:</p>
            <div style="text-align: center;">
              <a href="${verifyUrl}" class="button">Email Adresimi Doğrula</a>
            </div>
            <p>Veya aşağıdaki linki tarayıcınıza kopyalayabilirsiniz:</p>
            <p style="word-break: break-all; color: #667eea;">${verifyUrl}</p>
            <p><strong>Bu link 24 saat geçerlidir.</strong></p>
            <p>Email doğrulandıktan sonra, hesabınız admin onayı için gönderilecektir.</p>
          </div>
          <div class="footer">
            <p>Bu email'i siz talep etmediyseniz, lütfen görmezden gelin.</p>
            <p>&copy; 2025 ${APP_NAME}. Tüm hakları saklıdır.</p>
          </div>
        </div>
      </body>
    </html>
  `;

  return sendEmail({
    to: email,
    subject: `${APP_NAME} - Email Doğrulama`,
    html,
  });
}

/**
 * Send admin notification for new user
 */
export async function sendAdminNotification(userEmail: string, username: string, userId: string) {
  const adminEmail = process.env.ADMIN_EMAIL || 'admin@example.com';
  const approvalUrl = `${APP_URL}/admin/users?userId=${userId}`;

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #ff6b6b; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
          .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
          .button { display: inline-block; background: #ff6b6b; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
          .info { background: white; padding: 15px; border-left: 4px solid #ff6b6b; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🔔 Yeni Kullanıcı Kaydı</h1>
          </div>
          <div class="content">
            <h2>Admin Onayı Gerekiyor</h2>
            <p>Merhaba Admin,</p>
            <p>Yeni bir kullanıcı email doğrulamasını tamamladı ve onayınızı bekliyor.</p>
            <div class="info">
              <p><strong>Kullanıcı Adı:</strong> ${username}</p>
              <p><strong>Email:</strong> ${userEmail}</p>
              <p><strong>Kayıt Tarihi:</strong> ${new Date().toLocaleString('tr-TR')}</p>
            </div>
            <p>Kullanıcıyı onaylamak için admin paneline gidin:</p>
            <div style="text-align: center;">
              <a href="${approvalUrl}" class="button">Kullanıcıyı İncele</a>
            </div>
          </div>
          <div class="footer">
            <p>&copy; 2025 ${APP_NAME} Admin Panel</p>
          </div>
        </div>
      </body>
    </html>
  `;

  return sendEmail({
    to: adminEmail,
    subject: `${APP_NAME} - Yeni Kullanıcı Onayı Gerekiyor`,
    html,
  });
}

/**
 * Send user approval confirmation
 */
export async function sendApprovalConfirmation(email: string, username: string) {
  const loginUrl = `${APP_URL}/login`;

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
          .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
          .button { display: inline-block; background: #11998e; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>✅ Hesabınız Onaylandı!</h1>
          </div>
          <div class="content">
            <h2>Hoş Geldiniz ${username}!</h2>
            <p>Harika haber! Hesabınız admin tarafından onaylandı.</p>
            <p>Artık ${APP_NAME} platformunu kullanmaya başlayabilirsiniz. Premium özelliklere erişim için bir abonelik planı seçmeniz gerekmektedir.</p>
            <div style="text-align: center;">
              <a href="${loginUrl}" class="button">Giriş Yap</a>
            </div>
            <p>Sorularınız için bizimle iletişime geçebilirsiniz.</p>
          </div>
          <div class="footer">
            <p>&copy; 2025 ${APP_NAME}</p>
          </div>
        </div>
      </body>
    </html>
  `;

  return sendEmail({
    to: email,
    subject: `${APP_NAME} - Hesabınız Onaylandı!`,
    html,
  });
}

/**
 * Send password reset email
 */
export async function sendPasswordResetEmail(email: string, token: string) {
  const resetUrl = `${APP_URL}/reset-password?token=${token}`;

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: #f39c12; color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
          .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
          .button { display: inline-block; background: #f39c12; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🔐 Şifre Sıfırlama</h1>
          </div>
          <div class="content">
            <h2>Şifrenizi Sıfırlayın</h2>
            <p>Merhaba,</p>
            <p>Şifre sıfırlama talebinde bulundunuz. Aşağıdaki butona tıklayarak yeni şifrenizi oluşturabilirsiniz:</p>
            <div style="text-align: center;">
              <a href="${resetUrl}" class="button">Şifremi Sıfırla</a>
            </div>
            <p>Veya aşağıdaki linki tarayıcınıza kopyalayabilirsiniz:</p>
            <p style="word-break: break-all; color: #f39c12;">${resetUrl}</p>
            <p><strong>Bu link 1 saat geçerlidir.</strong></p>
            <p>Bu talebi siz yapmadıysanız, lütfen bu email'i görmezden gelin.</p>
          </div>
          <div class="footer">
            <p>&copy; 2025 ${APP_NAME}</p>
          </div>
        </div>
      </body>
    </html>
  `;

  return sendEmail({
    to: email,
    subject: `${APP_NAME} - Şifre Sıfırlama`,
    html,
  });
}

/**
 * Send payment confirmation email
 */
export async function sendPaymentConfirmation(email: string, plan: string, amount: number) {
  const dashboardUrl = `${APP_URL}/dashboard`;

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
          .container { max-width: 600px; margin: 0 auto; padding: 20px; }
          .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }
          .content { background: #f8f9fa; padding: 30px; border-radius: 0 0 10px 10px; }
          .button { display: inline-block; background: #667eea; color: white; padding: 15px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
          .info { background: white; padding: 15px; border-left: 4px solid #667eea; margin: 20px 0; }
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>💳 Ödeme Alındı!</h1>
          </div>
          <div class="content">
            <h2>Aboneliğiniz Aktif</h2>
            <p>Merhaba,</p>
            <p>Ödemeniz başarıyla alındı ve aboneliğiniz aktif edildi.</p>
            <div class="info">
              <p><strong>Plan:</strong> ${plan}</p>
              <p><strong>Tutar:</strong> ${amount} TL</p>
              <p><strong>Tarih:</strong> ${new Date().toLocaleString('tr-TR')}</p>
            </div>
            <p>Artık tüm premium özelliklere erişebilirsiniz!</p>
            <div style="text-align: center;">
              <a href="${dashboardUrl}" class="button">Dashboard'a Git</a>
            </div>
          </div>
          <div class="footer">
            <p>&copy; 2025 ${APP_NAME}</p>
          </div>
        </div>
      </body>
    </html>
  `;

  return sendEmail({
    to: email,
    subject: `${APP_NAME} - Ödeme Onayı`,
    html,
  });
}
