'use client';

/**
 * Registration Page - Premium Design
 *
 * White-hat compliance: Secure user registration interface
 */

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Mail,
  Lock,
  User,
  Loader2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  Eye,
  EyeOff,
  ArrowRight,
  Shield,
  Zap,
  Sparkles,
  Check
} from 'lucide-react';
import Link from 'next/link';

export default function RegisterPage() {
  const _router = useRouter();
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Client-side validation
    if (password !== confirmPassword) {
      setError('Şifreler eşleşmiyor');
      setLoading(false);
      return;
    }

    try {
      // Demo registration - save to localStorage
      const existingUser = localStorage.getItem('user-credentials');

      if (existingUser) {
        const { email: existingEmail } = JSON.parse(existingUser);
        if (existingEmail === email) {
          setError('Bu email adresi zaten kayıtlı');
          setLoading(false);
          return;
        }
      }

      // Save user credentials
      localStorage.setItem('user-credentials', JSON.stringify({
        email,
        username,
        password,
        registeredAt: new Date().toISOString(),
      }));

      setSuccess(true);
    } catch (error) {
      setError('Bir hata oluştu. Lütfen tekrar deneyin.');
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-950 via-black to-gray-900 text-white relative overflow-hidden pt-[100px]">
        {/* Animated Background */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 -left-32 w-96 h-96 bg-green-500/10 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
        </div>

        <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
          <div className="max-w-2xl w-full">
            <div className="bg-gray-900/80 backdrop-blur-xl rounded-3xl border border-gray-800/50 shadow-2xl shadow-green-500/10 p-8 md:p-12 text-center">
              <div className="w-20 h-20 bg-gradient-to-br from-green-600 to-emerald-600 rounded-full mx-auto mb-6 flex items-center justify-center shadow-lg shadow-green-500/20">
                <CheckCircle className="w-10 h-10 text-white" />
              </div>

              <h1 className="text-3xl md:text-4xl font-bold mb-4 bg-gradient-to-r from-green-400 to-emerald-400 text-transparent bg-clip-text">
                Kayıt Başarılı! 🎉
              </h1>

              <p className="text-gray-300 text-lg mb-8 max-w-md mx-auto">
                Emailinize bir doğrulama linki gönderdik. Lütfen emailinizi kontrol edin.
              </p>

              {/* Steps */}
              <div className="bg-blue-900/10 border border-blue-500/20 rounded-2xl p-6 mb-8 text-left max-w-md mx-auto">
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-blue-400" />
                  Sonraki Adımlar
                </h3>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="p-1 bg-blue-600/20 rounded-full mt-0.5">
                      <Check className="w-4 h-4 text-blue-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-200">Email Doğrulama</p>
                      <p className="text-xs text-gray-400">Email kutunuzu kontrol edin ve doğrulama linkine tıklayın</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="p-1 bg-purple-600/20 rounded-full mt-0.5">
                      <Shield className="w-4 h-4 text-purple-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-200">Admin Onayı</p>
                      <p className="text-xs text-gray-400">Admin hesabınızı inceleyip onaylayacak</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="p-1 bg-green-600/20 rounded-full mt-0.5">
                      <Zap className="w-4 h-4 text-green-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-200">Onay Emaili</p>
                      <p className="text-xs text-gray-400">Onay sonrası email ile bilgilendirileceksiniz</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="p-1 bg-cyan-600/20 rounded-full mt-0.5">
                      <TrendingUp className="w-4 h-4 text-cyan-400" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-200">Giriş ve Ödeme</p>
                      <p className="text-xs text-gray-400">Giriş yapıp plan seçerek tüm özelliklere erişin</p>
                    </div>
                  </div>
                </div>
              </div>

              <Link
                href="/login"
                className="inline-flex items-center gap-2 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white px-8 py-4 rounded-xl font-bold transition-all shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-105 active:scale-95"
              >
                Giriş Sayfasına Dön
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-black to-gray-900 text-white relative overflow-hidden pt-[100px]">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-32 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 -right-32 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-3xl" />
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex items-center justify-center min-h-screen p-4 py-12">
        <div className="w-full max-w-6xl grid md:grid-cols-2 gap-8 items-center">
          {/* Left Side - Branding */}
          <div className="hidden md:block space-y-8 px-8">
            <div className="space-y-4">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl shadow-lg shadow-blue-500/20">
                  <TrendingUp className="w-8 h-8" />
                </div>
                <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-500 text-transparent bg-clip-text">
                  LyTrade Scanner
                </h1>
              </div>
              <p className="text-xl text-gray-300 leading-relaxed">
                Kripto piyasalarında profesyonel sinyal analizi ve otomasyon platformuna katılın
              </p>
            </div>

            {/* Benefits */}
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-gray-200 mb-4">Platform Özellikleri:</h3>

              <div className="flex items-start gap-4 p-4 bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800/50 hover:border-blue-500/30 transition-all">
                <div className="p-2 bg-blue-600/20 rounded-lg">
                  <Zap className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h4 className="font-semibold mb-1">AI Destekli Sinyaller</h4>
                  <p className="text-sm text-gray-400">Yapay zeka ile gelişmiş piyasa analizi ve sinyal üretimi</p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800/50 hover:border-purple-500/30 transition-all">
                <div className="p-2 bg-purple-600/20 rounded-lg">
                  <Shield className="w-6 h-6 text-purple-400" />
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Güvenli API Entegrasyonu</h4>
                  <p className="text-sm text-gray-400">AES-256 şifreleme ile 5 farklı borsaya bağlanın</p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800/50 hover:border-cyan-500/30 transition-all">
                <div className="p-2 bg-cyan-600/20 rounded-lg">
                  <Sparkles className="w-6 h-6 text-cyan-400" />
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Otomatik İşlem Botu</h4>
                  <p className="text-sm text-gray-400">Kendi stratejinizle 24/7 otomatik trading</p>
                </div>
              </div>
            </div>
          </div>

          {/* Right Side - Register Form */}
          <div className="w-full max-w-md mx-auto">
            {/* Mobile Logo */}
            <div className="md:hidden text-center mb-8">
              <div className="inline-flex items-center gap-3 mb-4">
                <div className="p-2 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
                  <TrendingUp className="w-6 h-6" />
                </div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 text-transparent bg-clip-text">
                  LyTrade Scanner
                </h1>
              </div>
              <p className="text-gray-400">Yeni hesap oluşturun</p>
            </div>

            {/* Register Card */}
            <div className="bg-gray-900/80 backdrop-blur-xl rounded-3xl border border-gray-800/50 shadow-2xl shadow-purple-500/10 p-8">
              <div className="mb-6">
                <h2 className="text-2xl font-bold mb-2">Hemen Başlayın 🚀</h2>
                <p className="text-gray-400">Ücretsiz hesap oluşturun</p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Error Message */}
                {error && (
                  <div className="bg-red-900/20 border border-red-500/30 rounded-xl p-4 flex items-start gap-3 animate-in slide-in-from-top">
                    <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-red-300">{error}</p>
                  </div>
                )}

                {/* Email Field */}
                <div className="space-y-2">
                  <label htmlFor="email" className="block text-sm font-semibold text-gray-300">
                    Email Adresi
                  </label>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur" />
                    <Mail className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 group-hover:text-blue-400 transition-colors" />
                    <input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="relative w-full bg-gray-800/50 border border-gray-700/50 rounded-xl pl-12 pr-4 py-3.5 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50 transition-all placeholder:text-gray-500"
                      placeholder="ornek@email.com"
                    />
                  </div>
                </div>

                {/* Username Field */}
                <div className="space-y-2">
                  <label htmlFor="username" className="block text-sm font-semibold text-gray-300">
                    Kullanıcı Adı
                  </label>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur" />
                    <User className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 group-hover:text-purple-400 transition-colors" />
                    <input
                      id="username"
                      type="text"
                      value={username}
                      onChange={(e) => setUsername(e.target.value)}
                      required
                      minLength={3}
                      className="relative w-full bg-gray-800/50 border border-gray-700/50 rounded-xl pl-12 pr-4 py-3.5 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500/50 transition-all placeholder:text-gray-500"
                      placeholder="kullaniciadi"
                    />
                  </div>
                  <p className="text-xs text-gray-500">En az 3 karakter</p>
                </div>

                {/* Password Field */}
                <div className="space-y-2">
                  <label htmlFor="password" className="block text-sm font-semibold text-gray-300">
                    Şifre
                  </label>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-green-600 to-emerald-600 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur" />
                    <Lock className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 group-hover:text-green-400 transition-colors" />
                    <input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={8}
                      className="relative w-full bg-gray-800/50 border border-gray-700/50 rounded-xl pl-12 pr-12 py-3.5 focus:outline-none focus:ring-2 focus:ring-green-500/50 focus:border-green-500/50 transition-all placeholder:text-gray-500"
                      placeholder="••••••••••"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
                    >
                      {showPassword ? (
                        <EyeOff className="w-5 h-5" />
                      ) : (
                        <Eye className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                  <p className="text-xs text-gray-500">En az 8 karakter</p>
                </div>

                {/* Confirm Password Field */}
                <div className="space-y-2">
                  <label htmlFor="confirmPassword" className="block text-sm font-semibold text-gray-300">
                    Şifre Tekrar
                  </label>
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity blur" />
                    <Lock className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 group-hover:text-cyan-400 transition-colors" />
                    <input
                      id="confirmPassword"
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      required
                      className="relative w-full bg-gray-800/50 border border-gray-700/50 rounded-xl pl-12 pr-12 py-3.5 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all placeholder:text-gray-500"
                      placeholder="••••••••••"
                    />
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
                    >
                      {showConfirmPassword ? (
                        <EyeOff className="w-5 h-5" />
                      ) : (
                        <Eye className="w-5 h-5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={loading}
                  className="group relative w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-6 py-4 rounded-xl font-bold transition-all flex items-center justify-center gap-2 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-[0.98]"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Kayıt yapılıyor...
                    </>
                  ) : (
                    <>
                      Kayıt Ol
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </>
                  )}
                </button>
              </form>

              {/* Divider */}
              <div className="relative my-6">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-800" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-gray-900 text-gray-500">veya</span>
                </div>
              </div>

              {/* Login Link */}
              <div className="text-center">
                <p className="text-gray-400">
                  Zaten hesabınız var mı?{' '}
                  <Link
                    href="/login"
                    className="text-blue-400 hover:text-blue-300 font-semibold hover:underline transition-all"
                  >
                    Giriş yapın →
                  </Link>
                </p>
              </div>
            </div>

            {/* Terms */}
            <div className="mt-6 text-center text-xs text-gray-500">
              <p>
                Kayıt olarak{' '}
                <Link href="/terms" className="text-gray-400 hover:text-gray-300 underline">
                  Kullanım Koşulları
                </Link>
                {' '}ve{' '}
                <Link href="/privacy" className="text-gray-400 hover:text-gray-300 underline">
                  Gizlilik Politikası
                </Link>
                'nı kabul etmiş olursunuz.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
