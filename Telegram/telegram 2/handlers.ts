/**
 * 🎮 TELEGRAM BOT COMMAND HANDLERS
 * Kullanıcı komutlarını işle
 *
 * Features:
 * - /start - Welcome message
 * - /signals - Latest trading signals
 * - /price <SYMBOL> - Price query
 * - /help - Help menu
 * - Inline keyboard support
 * - White-hat compliant
 *
 * ⚠️ WHITE-HAT COMPLIANCE:
 * - Educational information only
 * - No trading operations
 * - Public data only
 * - User privacy protected
 */

import { bot } from './bot';
import { InlineKeyboard } from 'grammy';
import { isAllowedChatId, isPrivateMode } from './config';
import { subscribe } from './notifications';

// ============================================================================
// COMMAND HANDLERS
// ============================================================================

/**
 * /start - Hoş geldin mesajı
 * 🔒 Gizli mod aktifse sadece izin verilen kullanıcılar girebilir
 */
bot.command('start', async (ctx) => {
  const chatId = ctx.chat?.id;
  if (!chatId) {
    await ctx.reply('❌ Hata: Chat ID bulunamadı.');
    return;
  }

  // 🔒 GIZLI MOD KONTROLÜ
  if (isPrivateMode() && !isAllowedChatId(chatId)) {
    await ctx.reply(
      `🔒 **Bu bot gizli moddadır**

Bu bot sadece yetkili kullanıcılar tarafından kullanılabilir.

Chat ID: \`${chatId}\`

Bot sahibiyseniz bu Chat ID'yi .env dosyasına ekleyin:
\`TELEGRAM_ALLOWED_CHAT_IDS=${chatId}\``,
      { parse_mode: 'Markdown' }
    );
    return;
  }

  // Abone et
  const subscribed = subscribe(chatId);

  const keyboard = new InlineKeyboard()
    .text('📊 Son Sinyaller', 'signals')
    .row()
    .text('💰 Fiyat Sorgula', 'price')
    .row()
    .text('❓ Yardım', 'help')
    .url('🌐 Web Sitesi', process.env.NEXT_PUBLIC_APP_URL || 'https://lydian.app');

  await ctx.reply(
    `👋 Hoş geldin, ${ctx.from?.first_name || 'Trader'}!

🤖 **LyTrade Trading Scanner Bot**
${isPrivateMode() ? '🔒 **(Gizli Mod Aktif)**' : ''}

${subscribed ? '✅ Bildirimler aktif edildi!' : '⚠️ Bildirim aktivasyonu başarısız.'}

Bu bot ile:
✅ Gerçek zamanlı trading sinyalleri al
✅ Fiyat bilgilerini sorgula
✅ Piyasa analizlerini takip et

**⚠️ Önemli Uyarı:**
Bu bot sadece eğitim amaçlıdır. Finansal tavsiye değildir.

Komutlar:
/signals - Son trading sinyalleri
/price BTCUSDT - Fiyat sorgula
/help - Yardım menüsü`,
    {
      parse_mode: 'Markdown',
      reply_markup: keyboard,
    }
  );
});

/**
 * /signals - Son trading sinyalleri
 */
bot.command('signals', async (ctx) => {
  try {
    await ctx.reply('⏳ Sinyaller yükleniyor...');

    // LyTrade API'den sinyalleri çek
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
    const response = await fetch(`${baseUrl}/api/signals`);

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.signals || data.signals.length === 0) {
      await ctx.reply('⚠️ Şu anda aktif sinyal yok.');
      return;
    }

    // İlk 10 sinyali formatla
    const messages = data.signals.slice(0, 10).map((signal: any, i: number) => {
      const icon = signal.action === 'BUY' ? '🟢' : '🔴';
      const emoji = signal.action === 'BUY' ? '📈' : '📉';

      return `${icon} **${i + 1}. ${signal.symbol}**
💰 Fiyat: $${signal.price}
${emoji} Sinyal: **${signal.action}**
🎯 Güven: ${signal.confidence}%
⏰ ${new Date(signal.timestamp).toLocaleString('tr-TR')}`;
    });

    const keyboard = new InlineKeyboard().url(
      '🌐 Detaylı Görüntüle',
      `${process.env.NEXT_PUBLIC_APP_URL || 'https://lydian.app'}/trading-signals`
    );

    await ctx.reply(
      `📊 **SON 10 TRADİNG SİNYALİ**

${messages.join('\n\n---\n\n')}

⚠️ *Eğitim amaçlıdır, finansal tavsiye değildir.*`,
      {
        parse_mode: 'Markdown',
        reply_markup: keyboard,
      }
    );
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram /signals] Error:', error);
    }

    await ctx.reply('❌ Sinyaller yüklenirken hata oluştu. Lütfen daha sonra tekrar deneyin.');
  }
});

/**
 * /price <SYMBOL> - Fiyat sorgula
 */
bot.command('price', async (ctx) => {
  const symbol = ctx.match?.toString().toUpperCase().trim() || 'BTCUSDT';

  try {
    await ctx.reply(`⏳ ${symbol} fiyatı sorgulanıyor...`);

    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';
    const response = await fetch(`${baseUrl}/api/binance/futures?symbols=${symbol}`);

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.data || data.data.length === 0) {
      await ctx.reply(`⚠️ ${symbol} için fiyat bulunamadı.`);
      return;
    }

    const coin = data.data[0];
    const priceChange = parseFloat(coin.priceChangePercent);
    const changeIcon = priceChange > 0 ? '📈' : '📉';
    const changeColor = priceChange > 0 ? '🟢' : '🔴';

    const keyboard = new InlineKeyboard().url(
      '📊 Analiz Görüntüle',
      `${process.env.NEXT_PUBLIC_APP_URL || 'https://lydian.app'}/trading-signals`
    );

    await ctx.reply(
      `💰 **${coin.symbol}**

Fiyat: **$${parseFloat(coin.lastPrice).toFixed(2)}**
${changeIcon} 24s Değişim: ${changeColor} ${coin.priceChangePercent}%
📊 Hacim: $${parseFloat(coin.quoteVolume).toLocaleString('tr-TR')}
⏰ ${new Date().toLocaleString('tr-TR')}

⚠️ *Eğitim amaçlıdır, finansal tavsiye değildir.*`,
      {
        parse_mode: 'Markdown',
        reply_markup: keyboard,
      }
    );
  } catch (error: any) {
    const nodeEnv = process.env.NODE_ENV as string;
    if (nodeEnv !== 'production') {
      console.error('[Telegram /price] Error:', error);
    }

    await ctx.reply('❌ Fiyat sorgulanırken hata oluştu. Lütfen daha sonra tekrar deneyin.');
  }
});

/**
 * /help - Yardım menüsü
 */
bot.command('help', async (ctx) => {
  const keyboard = new InlineKeyboard()
    .text('📊 Sinyaller', 'signals')
    .text('💰 Fiyat', 'price')
    .row()
    .url('🌐 Web Sitesi', process.env.NEXT_PUBLIC_APP_URL || 'https://lydian.app');

  await ctx.reply(
    `❓ **YARDIM MENÜSÜ**

**Temel Komutlar:**
/start - Başlangıç menüsü
/signals - Son 10 trading sinyali
/price <SYMBOL> - Fiyat sorgula
  Örnek: /price BTCUSDT
  Örnek: /price ETHUSDT

/help - Bu yardım menüsü

**Önemli Bilgiler:**
⚠️ Bu bot sadece eğitim amaçlıdır
⚠️ Finansal tavsiye değildir
⚠️ Gerçek para ile işlem yapmayın
⚠️ Kendi araştırmanızı yapın

🌐 Web: ${process.env.NEXT_PUBLIC_APP_URL || 'https://lydian.app'}`,
    {
      parse_mode: 'Markdown',
      reply_markup: keyboard,
    }
  );
});

// ============================================================================
// INLINE KEYBOARD CALLBACKS
// ============================================================================

/**
 * Callback: signals
 */
bot.callbackQuery('signals', async (ctx) => {
  await ctx.answerCallbackQuery();

  // /signals komutunu tetikle
  const signalsCommand = bot.command('signals');
  if (signalsCommand) {
    await ctx.reply('📊 Sinyaller yükleniyor...');
    // Trigger signals command manually
    const fakeCtx = {
      ...ctx,
      match: '',
    };
    // Call the signals handler
    const handler = (bot as any).handlers.find((h: any) =>
      h.triggers && h.triggers.some((t: any) => t === 'signals')
    );
    if (handler) {
      await handler.middleware(fakeCtx);
    }
  }
});

/**
 * Callback: price
 */
bot.callbackQuery('price', async (ctx) => {
  await ctx.answerCallbackQuery();
  await ctx.reply('💰 Hangi coin için fiyat sorgulayalım?\n\nÖrnek: /price BTCUSDT');
});

/**
 * Callback: help
 */
bot.callbackQuery('help', async (ctx) => {
  await ctx.answerCallbackQuery();

  // Trigger help command
  const helpCommand = (bot as any).handlers.find((h: any) =>
    h.triggers && h.triggers.some((t: any) => t === 'help')
  );
  if (helpCommand) {
    await helpCommand.middleware(ctx);
  }
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

/**
 * Global error handler
 */
bot.catch((err) => {
  const nodeEnv = process.env.NODE_ENV as string;
  if (nodeEnv !== 'production') {
    console.error('[Telegram Bot] Error:', err);
  }
});

export default bot;
