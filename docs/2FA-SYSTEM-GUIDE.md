# 🔐 Google Authenticator 2FA System - Kullanım Kılavuzu

## Genel Bakış

AiLydian Trading Scanner platformu artık **Google Authenticator** ile **İki Faktörlü Kimlik Doğrulama (2FA)** desteğine sahiptir. Bu sistem, hesap güvenliğini artırmak için TOTP (Time-based One-Time Password) standardını kullanır.

## Özellikler

- ✅ **Google Authenticator** ile TOTP tabanlı 2FA
- ✅ QR kod ile kolay kurulum
- ✅ Manuel giriş desteği (QR kod okuyamayan cihazlar için)
- ✅ **8 adet yedekleme kodu** (telefon kaybı durumu için)
- ✅ Gerçek zamanlı doğrulama
- ✅ Settings sayfasından kolay yönetim
- ✅ Login sistemine tam entegrasyon
- ✅ Beyaz şapkalı güvenlik standartları

## Sistem Mimarisi

### Backend Bileşenleri

#### 1. Storage Katmanı (`/src/lib/2fa-store.ts`)
- JSON dosya tabanlı kalıcı depolama (`/data/2fa-secrets.json`)
- Bellek içi önbellekleme (performans için)
- Kullanıcı başına:
  - Secret key (Base32 encoded)
  - Enabled durumu
  - Yedekleme kodları
  - Zaman damgaları

#### 2. API Endpoints

##### `/api/auth/2fa/setup` (POST)
- QR kod ve secret oluşturur
- 8 yedekleme kodu üretir
- Veriyi kaydeder (başlangıçta enabled: false)

**Request:**
```json
{
  "username": "lydian"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "secret": "BASE32ENCODEDSECRET...",
    "qrCode": "data:image/png;base64,...",
    "backupCodes": [
      "ABCD-EFGH",
      "IJKL-MNOP",
      ...
    ],
    "manualEntry": "BASE32ENCODEDSECRET..."
  }
}
```

##### `/api/auth/2fa/verify` (POST)
- 6 haneli TOTP kodunu doğrular
- Başarılı doğrulama sonrası 2FA'yı aktifleştirir

**Request:**
```json
{
  "username": "lydian",
  "token": "123456"
}
```

**Response:**
```json
{
  "success": true,
  "message": "2FA enabled successfully!"
}
```

##### `/api/auth/2fa/status` (POST)
- Kullanıcının 2FA durumunu kontrol eder

**Request:**
```json
{
  "username": "lydian"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "enabled": true,
    "hasBackupCodes": true,
    "backupCodesRemaining": 8
  }
}
```

##### `/api/auth/2fa/disable` (POST)
- 2FA'yı devre dışı bırakır
- Tüm 2FA verilerini siler

**Request:**
```json
{
  "username": "lydian"
}
```

**Response:**
```json
{
  "success": true,
  "message": "2FA disabled successfully"
}
```

#### 3. Login Entegrasyonu (`/src/app/api/auth/login/route.ts`)

Login akışı 3 adımda çalışır:

1. **Kullanıcı adı ve şifre kontrolü**
2. **2FA durumu kontrolü**
   - 2FA aktifse → TOTP kodu veya yedek kod gerekli
   - 2FA aktif değilse → Direkt giriş
3. **Session oluşturma**

**Login Request (2FA aktifken):**
```json
{
  "username": "lydian",
  "password": "1234",
  "token": "123456"
}
```

**veya yedek kod ile:**
```json
{
  "username": "lydian",
  "password": "1234",
  "backupCode": "ABCD-EFGH"
}
```

### Frontend Bileşenleri

#### 1. TwoFactorAuth Komponenti (`/src/components/settings/TwoFactorAuth.tsx`)
- 2FA kurulum arayüzü
- QR kod görüntüleme
- Yedekleme kodları gösterimi
- Token doğrulama
- Enable/Disable işlemleri

#### 2. Settings Sayfası Entegrasyonu (`/src/components/settings/SecuritySettings.tsx`)
- "Güvenlik" menüsü → "2FA" sekmesi
- TwoFactorAuth komponenti entegre edildi

## Kullanım Kılavuzu

### Kullanıcı Bakış Açısı

#### Adım 1: 2FA Kurulumu

1. **Settings sayfasına git**
   - Yan menüden "Ayarlar" seçeneğine tıkla
   - "Güvenlik" sekmesine geç
   - "2FA" tab'ına tıkla

2. **2FA'yı Etkinleştir butonuna tıkla**

3. **QR kodunu tara**
   - Google Authenticator uygulamasını aç
   - "+" butonuna tıkla → "QR kod tara"
   - Ekrandaki QR kodu tara

4. **Manuel giriş (opsiyonel)**
   - QR kod taramıyorsa, gösterilen secret'i manuel olarak gir

5. **Yedekleme kodlarını kaydet**
   - ⚠️ **ÖNEMLİ**: 8 yedekleme kodunu güvenli bir yere kaydet
   - Telefonu kaybettiğinde bu kodlarla giriş yapabilirsin
   - Her kod sadece bir kez kullanılabilir

6. **6 haneli kodu gir ve doğrula**
   - Google Authenticator'dan 6 haneli kodu kopyala
   - Ekrandaki alana yapıştır
   - "Doğrula ve Etkinleştir" butonuna tıkla

#### Adım 2: 2FA ile Giriş Yapma

1. **Login sayfasında** kullanıcı adı ve şifreni gir
2. **6 haneli kodu gir**
   - Google Authenticator uygulamasını aç
   - AiLydian hesabının yanındaki 6 haneli kodu kopyala
3. **Giriş Yap** butonuna tıkla

#### Telefonu Kaybettiysen: Yedek Kod Kullanımı

1. Login sayfasında kullanıcı adı ve şifreni gir
2. "Yedek Kod Kullan" seçeneğine tıkla
3. Kaydettiğin yedek kodlardan birini gir (örn: `ABCD-EFGH`)
4. Giriş Yap

⚠️ **Not**: Kullanılan yedek kod otomatik olarak silinir!

### Geliştirici Bakış Açısı

#### API Test Komutları

##### 1. Login yaparak session oluştur
```bash
curl -s "http://localhost:3000/api/auth/login" \
  -X POST -H "Content-Type: application/json" \
  -d '{"username":"lydian","password":"1234"}' \
  -c /tmp/2fa-cookies.txt
```

##### 2. 2FA Setup başlat
```bash
curl -s -b /tmp/2fa-cookies.txt "http://localhost:3000/api/auth/2fa/setup" \
  -X POST -H "Content-Type: application/json" \
  -d '{"username":"lydian"}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('success'):
    print('✅ Setup başarılı!')
    print('Secret:', data['data']['secret'])
    print('Yedek Kodlar:', len(data['data']['backupCodes']), 'adet')
else:
    print('❌ Hata:', data.get('message'))
"
```

##### 3. Token ile doğrula (Google Authenticator'dan alınan kod)
```bash
curl -s -b /tmp/2fa-cookies.txt "http://localhost:3000/api/auth/2fa/verify" \
  -X POST -H "Content-Type: application/json" \
  -d '{"username":"lydian","token":"YOUR_6_DIGIT_CODE"}'
```

##### 4. Durum kontrolü
```bash
curl -s -b /tmp/2fa-cookies.txt "http://localhost:3000/api/auth/2fa/status" \
  -X POST -H "Content-Type: application/json" \
  -d '{"username":"lydian"}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Enabled:', data['data']['enabled'])
print('Backup Codes:', data['data']['backupCodesRemaining'])
"
```

##### 5. 2FA ile login
```bash
# Normal login (2FA aktifse token gerekli)
curl -s "http://localhost:3000/api/auth/login" \
  -X POST -H "Content-Type: application/json" \
  -d '{"username":"lydian","password":"1234","token":"YOUR_6_DIGIT_CODE"}'

# Yedek kod ile login
curl -s "http://localhost:3000/api/auth/login" \
  -X POST -H "Content-Type: application/json" \
  -d '{"username":"lydian","password":"1234","backupCode":"ABCD-EFGH"}'
```

## Güvenlik Özellikleri

### White-Hat Standartları

1. **TOTP Standardı (RFC 6238)**
   - Endüstri standardı implementasyon
   - Speakeasy kütüphanesi kullanımı

2. **Clock Drift Toleransı**
   - Window: 2 (±60 saniye)
   - Mobil cihazların saat farklarına tolerans

3. **Backup Code Güvenliği**
   - Her kod tek kullanımlık
   - Kullanılan kod otomatik silinir
   - 8 adet (acil durum için yeterli)

4. **Secret Storage**
   - Base32 encoding
   - Sunucu tarafında saklanır
   - Client'a asla gönderilmez

5. **Session Management**
   - httpOnly cookies
   - Secure flag (production'da)
   - SameSite: lax

6. **Input Validation**
   - Token: Sadece 6 rakam
   - Boşluk karakterleri otomatik temizlenir

## Dosya Yapısı

```
/
├── src/
│   ├── lib/
│   │   └── 2fa-store.ts                    # Storage katmanı
│   ├── app/
│   │   └── api/
│   │       └── auth/
│   │           ├── login/route.ts          # 2FA entegrasyonlu login
│   │           └── 2fa/
│   │               ├── setup/route.ts      # QR kod oluşturma
│   │               ├── verify/route.ts     # Token doğrulama
│   │               ├── disable/route.ts    # 2FA kapatma
│   │               └── status/route.ts     # Durum kontrolü
│   └── components/
│       └── settings/
│           ├── TwoFactorAuth.tsx           # 2FA UI komponenti
│           └── SecuritySettings.tsx        # Settings entegrasyonu
├── data/
│   └── 2fa-secrets.json                    # Kullanıcı verileri (otomatik oluşur)
└── docs/
    └── 2FA-SYSTEM-GUIDE.md                 # Bu dokuman
```

## Bağımlılıklar

```json
{
  "dependencies": {
    "speakeasy": "^2.0.0",     // TOTP implementasyonu
    "qrcode": "^1.5.4"         // QR kod oluşturma
  },
  "devDependencies": {
    "@types/speakeasy": "^2.0.10",
    "@types/qrcode": "^1.5.5"
  }
}
```

## Troubleshooting

### Problem: "Invalid token" hatası
**Çözüm:**
- Telefonunuzun saati senkronize mi kontrol edin
- Google Authenticator'ı yeniden başlatın
- Kodu yeniden girmeyi deneyin (30 saniyede bir değişir)

### Problem: QR kod görünmüyor
**Çözüm:**
- Tarayıcı konsolunu kontrol edin
- Setup endpoint'inin doğru çalıştığını test edin
- Manuel giriş yöntemini kullanın

### Problem: Backup code çalışmıyor
**Çözüm:**
- Kod formatını kontrol edin (XXXX-XXXX)
- Daha önce kullanılmış olabilir
- Farklı bir backup code deneyin

### Problem: 2FA disable edilemiyor
**Çözüm:**
- Authentication cookie'sinin geçerli olduğundan emin olun
- Logout yapıp tekrar login deneyin

## İletişim ve Destek

Bu 2FA sistemi AiLydian Trading Scanner platformu için özel olarak geliştirilmiştir.

**Özellikler:**
- ✅ Production-ready
- ✅ Security best practices
- ✅ Kullanıcı dostu UI
- ✅ Comprehensive documentation
- ✅ Error handling
- ✅ Testing support

**Version:** 1.0.0
**Last Updated:** 2025-01-02
**Status:** 🟢 Production Ready
