'use client';

/**
 * Email Verification Page
 *
 * White-hat compliance: User-friendly email verification interface
 */

import { useEffect, useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { CheckCircle, XCircle, Loader2 } from 'lucide-react';

function VerifyEmailContent() {
  const searchParams = useSearchParams();
  const token = searchParams.get('token');

  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading');
  const [message, setMessage] = useState('');

  useEffect(() => {
    if (!token) {
      setStatus('error');
      setMessage('Doğrulama token\'ı bulunamadı');
      return;
    }

    verifyEmail(token);
  }, [token]);

  const verifyEmail = async (token: string) => {
    try {
      const response = await fetch(`/api/auth/verify-email?token=${token}`);
      const data = await response.json();

      if (response.ok) {
        setStatus('success');
        setMessage(data.message);
      } else {
        setStatus('error');
        setMessage(data.error || 'Doğrulama başarısız');
      }
    } catch (error) {
      setStatus('error');
      setMessage('Bir hata oluştu. Lütfen tekrar deneyin.');
    }
  };

  return (
    <div className="min-h-screen bg-black text-white flex items-center justify-center p-4">
      <div className="max-w-md w-full">
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-8 text-center">
          {status === 'loading' && (
            <>
              <Loader2 className="w-16 h-16 text-blue-500 mx-auto mb-4 animate-spin" />
              <h1 className="text-2xl font-bold mb-2">Email Doğrulanıyor...</h1>
              <p className="text-gray-400">Lütfen bekleyin</p>
            </>
          )}

          {status === 'success' && (
            <>
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h1 className="text-2xl font-bold mb-2">Email Doğrulandı!</h1>
              <p className="text-gray-400 mb-6">{message}</p>
              <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4 mb-6">
                <p className="text-sm text-blue-300">
                  ✅ Email adresiniz doğrulandı<br />
                  ⏳ Admin onayı bekleniyor<br />
                  📧 Onay sonrası email alacaksınız
                </p>
              </div>
              <a
                href="/login"
                className="inline-block bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-6 py-3 rounded-lg font-semibold transition-all"
              >
                Giriş Sayfasına Dön
              </a>
            </>
          )}

          {status === 'error' && (
            <>
              <XCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h1 className="text-2xl font-bold mb-2">Doğrulama Başarısız</h1>
              <p className="text-gray-400 mb-6">{message}</p>
              <div className="space-y-3">
                <a
                  href="/register"
                  className="block bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-all"
                >
                  Yeni Kayıt Oluştur
                </a>
                <a
                  href="/login"
                  className="block bg-gray-800 hover:bg-gray-700 text-white px-6 py-3 rounded-lg font-semibold transition-all"
                >
                  Giriş Sayfasına Dön
                </a>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default function VerifyEmailPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-black text-white flex items-center justify-center p-4">
        <div className="max-w-md w-full">
          <div className="bg-gray-900 rounded-2xl border border-gray-800 p-8 text-center">
            <Loader2 className="w-16 h-16 text-blue-500 mx-auto mb-4 animate-spin" />
            <h1 className="text-2xl font-bold mb-2">Yükleniyor...</h1>
            <p className="text-gray-400">Lütfen bekleyin</p>
          </div>
        </div>
      </div>
    }>
      <VerifyEmailContent />
    </Suspense>
  );
}
