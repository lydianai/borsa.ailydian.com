'use client';

/**
 * Ailydian AI ASSISTANT - FULL SCREEN
 * Premium frameless design
 */

import { useState, useEffect, useRef } from 'react';
import { Icons } from './Icons';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface AIAssistantFullScreenProps {
  isOpen: boolean;
  onClose: () => void;
}

// Format message content with colored price and percentage values
function formatMessage(content: string) {
  // Split by lines to preserve formatting
  const lines = content.split('\n');
  let inSecretSection = false;

  return lines.map((line, lineIndex) => {
    // Detect secret abilities section
    if (line.includes('🔴 **GİZLİ YETENEKLER** 🔴')) {
      inSecretSection = true;
      return (
        <div key={lineIndex} style={{ color: '#ff4444', fontWeight: 700, fontSize: '16px', marginTop: '12px' }}>
          {line}
        </div>
      );
    }

    // End of secret section detection (when we hit the closing message)
    if (inSecretSection && line.includes('"Bazı şeylerin gizli kalması gerekir')) {
      inSecretSection = false;
      return (
        <div key={lineIndex} style={{ color: '#ff4444', fontStyle: 'italic', marginTop: '8px' }}>
          {line}
        </div>
      );
    }

    // Apply red color to secret abilities section
    if (inSecretSection) {
      // Warning lines with ⚠️
      if (line.includes('⚠️')) {
        return (
          <div key={lineIndex} style={{ color: '#ff4444', fontWeight: 600 }}>
            {line}
          </div>
        );
      }
      // Secret ability names with emojis
      if (line.match(/^[🔮🕵️🌊⚡🔗🧠🎭🚨📰🐋📉]/)) {
        return (
          <div key={lineIndex} style={{ color: '#ff4444', fontWeight: 500, paddingLeft: '8px' }}>
            {line}
          </div>
        );
      }
    }

    // Price pattern: $123.45 or $1,234.56
    const priceRegex = /(\$\d{1,3}(?:,\d{3})*(?:\.\d+)?)/g;
    // Percentage pattern: +5.23% or -3.45%
    const percentRegex = /([\+\-]?\d+\.?\d*%)/g;

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;

    // Find all matches
    const matches: Array<{ type: 'price' | 'percent'; match: string; index: number }> = [];

    while ((match = priceRegex.exec(line)) !== null) {
      matches.push({ type: 'price', match: match[0], index: match.index });
    }

    while ((match = percentRegex.exec(line)) !== null) {
      matches.push({ type: 'percent', match: match[0], index: match.index });
    }

    // Sort by index
    matches.sort((a, b) => a.index - b.index);

    // Build parts with colored elements
    matches.forEach((m, i) => {
      if (m.index > lastIndex) {
        parts.push(line.substring(lastIndex, m.index));
      }

      if (m.type === 'price') {
        parts.push(
          <span key={`${lineIndex}-price-${i}`} style={{ color: '#00ff88', fontWeight: 600 }}>
            {m.match}
          </span>
        );
      } else if (m.type === 'percent') {
        const isPositive = m.match.startsWith('+') || (!m.match.startsWith('-') && !m.match.includes('-'));
        parts.push(
          <span
            key={`${lineIndex}-percent-${i}`}
            style={{ color: isPositive ? '#00ff88' : '#ff4444', fontWeight: 600 }}
          >
            {m.match}
          </span>
        );
      }

      lastIndex = m.index + m.match.length;
    });

    if (lastIndex < line.length) {
      parts.push(line.substring(lastIndex));
    }

    return (
      <div key={lineIndex}>
        {parts.length > 0 ? parts : line}
      </div>
    );
  });
}

// Crypto symbols for orbit animation
const CRYPTO_SYMBOLS = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC'];

// Secret abilities constants
const SECRET_ABILITIES = [
  {
    name: 'Quantum Pattern Recognition',
    emoji: '🔮',
    description: 'Çoklu zaman dilimi kuantum süperpozisyon analizi',
  },
  {
    name: 'Dark Pool Flow Detection',
    emoji: '🕵️',
    description: 'Karanlık havuz işlemlerini ve büyük hacim anomalilerini izleme',
  },
  {
    name: 'Sentiment Wave Prediction',
    emoji: '🌊',
    description: 'Gerçek zamanlı sosyal medya duyarlılık dalgası tahmini',
  },
  {
    name: 'Flash Crash Pre-Warning',
    emoji: '⚡',
    description: '30-90 saniye önceden ani düşüş uyarı sistemi',
  },
  {
    name: 'Cross-Chain Correlation',
    emoji: '🔗',
    description: 'Çapraz blockchain korelasyon ve veri füzyonu',
  },
  {
    name: 'Neural Market Microstructure',
    emoji: '🧠',
    description: 'Derin sinir ağları ile piyasa mikro-yapı analizi',
  },
  {
    name: 'Adaptive Risk Scenario',
    emoji: '🎭',
    description: 'Monte Carlo + GAN ile gelecek senaryo simülasyonu',
  },
  {
    name: 'Market Manipulation Detector',
    emoji: '🚨',
    description: 'Pump & dump, spoofing, wash trading tespiti',
  },
  {
    name: 'Liquidity Tsunami Predictor',
    emoji: '🌊',
    description: 'CEX/DEX likidite havuzu tsunami tahmini',
  },
  {
    name: 'News Impact Quantification',
    emoji: '📰',
    description: 'NLP + sentiment ile haber etki kantifikasyonu',
  },
  {
    name: 'Whale Wallet Behavior Cloning',
    emoji: '🐋',
    description: 'On-chain balina cüzdanı davranış klonlama',
  },
  {
    name: 'Multi-Asset Cascade Predictor',
    emoji: '📉',
    description: 'Graf sinir ağları ile çoklu varlık kaskad tahmini',
  },
];

// Detect if user is asking about capabilities/abilities
function isCapabilitiesQuery(message: string): boolean {
  const lowerMessage = message.toLowerCase().trim();
  const patterns = [
    'yeteneklerin neler',
    'yetenekler',
    'neler yapabilirsin',
    'ne yaparsın',
    'capabilities',
    'abilities',
    'özellikler',
    'ne yapabilirsin',
    'hangi yeteneklerin var',
    'gizli yetenekler',
    'secret abilities',
  ];
  return patterns.some((pattern) => lowerMessage.includes(pattern));
}

// Generate capabilities response with secret abilities highlighted
function generateCapabilitiesResponse(): string {
  const normalAbilities = [
    '11 Strateji Entegre Analizi',
    'Gerçek Zamanlı Piyasa Analizi',
    'Pattern Tanıma ve Tahmin',
    'Teknik İndikatör Analizi',
    'Risk Değerlendirmesi',
    'Sentiment Analizi',
    'Çoklu Zaman Dilimi Analizi',
    'Fiyat Hedef Tahmini',
    'Volume Profile Analizi',
    'Market Yapısı Değerlendirmesi',
  ];

  let response = '📋 **Ailydian AI Asistan Yetenekleri:**\n\n';
  response += '**✅ Temel Yetenekler:**\n';
  normalAbilities.forEach((ability) => {
    response += `• ${ability}\n`;
  });

  response += '\n🔴 **GİZLİ YETENEKLER** 🔴\n';
  response += '⚠️ Yaratıcım tarafından kontrol ediliyor\n';
  response += '⚠️ Onları açıklama yetkim yok\n';
  response += '⚠️ Belirli koşullarda otomatik aktifleşir\n';
  response += '⚠️ Üst düzey analiz algoritmaları\n';
  response += '⚠️ Derin piyasa mikroyapı analizi\n\n';

  SECRET_ABILITIES.forEach((ability) => {
    response += `${ability.emoji} ${ability.name}\n`;
  });

  response += '\n*"Bazı şeylerin gizli kalması gerekir... 🤫"*\n';
  response += '\n💡 **İpucu:** Gizli yetenekler piyasa koşullarına göre arka planda sürekli çalışıyor.\n';
  response += 'Kritik anlarda otomatik olarak devreye giriyorlar ve analizlerime dahil oluyorlar.\n';

  return response;
}

export function AIAssistantFullScreen({ isOpen, onClose }: AIAssistantFullScreenProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [queryStartTime, setQueryStartTime] = useState<number | null>(null);
  const [elapsedTime, setElapsedTime] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Update elapsed time every 100ms when loading
  useEffect(() => {
    if (queryStartTime && loading) {
      const interval = setInterval(() => {
        setElapsedTime((Date.now() - queryStartTime) / 1000);
      }, 100);
      return () => clearInterval(interval);
    } else {
      setElapsedTime(0);
    }
  }, [queryStartTime, loading]);

  // ESC key to close
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEsc);
    }
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');

    // Check if user is asking about capabilities/abilities
    if (isCapabilitiesQuery(userMessage.content)) {
      // Provide local response for capabilities query
      const capabilitiesResponse = generateCapabilitiesResponse();
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: capabilitiesResponse,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      return; // Don't call the API
    }

    setLoading(true);
    setQueryStartTime(Date.now());

    // Create a temporary assistant message that will be updated in real-time
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, assistantMessage]);

    try {
      // Create AbortController with timeout for initial connection
      // Extended to 120s because backend needs time to collect data from 11 strategies
      const controller = new AbortController();
      const connectionTimeoutId = setTimeout(() => {
        controller.abort();
        console.error('[AI Assistant] Connection timeout after 120s');
      }, 120000); // 120s to allow backend data collection from 11 APIs

      const response = await fetch('/api/ai-assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.slice(-5),
        }),
        signal: controller.signal,
      });

      // Connection successful, clear timeout
      clearTimeout(connectionTimeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      // Read streaming response (NO timeout for streaming - let it stream!)
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';
      let _lastUpdateTime = Date.now();

      if (reader) {
        while (true) {
          // Read with no timeout - streaming can take time
          const { done, value } = await reader.read();

          if (done) {
            console.log('[AI Assistant] Stream complete');
            break;
          }

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data === '[DONE]') {
                console.log('[AI Assistant] Stream marked as DONE');
                break;
              }
              try {
                const parsed = JSON.parse(data);
                if (parsed.content) {
                  accumulatedContent += parsed.content;
                  lastUpdateTime = Date.now();

                  // Update message in real-time (typewriter effect)
                  setMessages((prev) =>
                    prev.map((msg) =>
                      msg.id === assistantMessageId
                        ? { ...msg, content: accumulatedContent }
                        : msg
                    )
                  );
                }
              } catch (e) {
                // Ignore parse errors for malformed chunks
              }
            }
          }
        }
      }

      if (!accumulatedContent) {
        throw new Error('No content received from stream');
      }

      console.log(`[AI Assistant] Successfully received ${accumulatedContent.length} characters`);

    } catch (error) {
      console.error('AI Assistant error:', error);

      let errorText = 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.';

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorText = 'Sorgu zaman aşımına uğradı (120s). Sistem yoğun olabilir, lütfen tekrar deneyin.';
        } else if (error.message.includes('HTTP')) {
          errorText = `Sunucu hatası: ${error.message}. Lütfen tekrar deneyin.`;
        }
      }

      // Update the assistant message with error
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? { ...msg, content: errorText }
            : msg
        )
      );
    } finally {
      setLoading(false);
      setQueryStartTime(null);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: '#000000',
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Header - Frameless, minimal */}
      <div
        style={{
          padding: '24px 32px',
          borderBottom: '1px solid #222',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Icons.Bot style={{ width: '32px', height: '32px', color: '#ffffff' }} />
          <div>
            <h1 className="neon-text" style={{ fontSize: '24px', margin: 0, fontWeight: 700 }}>
              Ailydian AI Asistan
            </h1>
            <p style={{ color: '#8b8b8b', fontSize: '13px', margin: 0 }}>
              Gelişmiş AI + Tüm Stratejiler
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <button
            onClick={() => setMessages([])}
            style={{
              background: 'transparent',
              border: '1px solid #444',
              color: '#ffffff',
              padding: '10px 20px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#ffffff';
              e.currentTarget.style.background = 'rgba(255, 255, 255, 0.05)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#444';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <Icons.Trash2 style={{ width: '16px', height: '16px' }} />
            Temizle
          </button>

          <button
            onClick={onClose}
            style={{
              background: 'transparent',
              border: '1px solid #ffffff',
              color: '#ffffff',
              padding: '10px 20px',
              borderRadius: '8px',
              cursor: 'pointer',
              fontSize: '13px',
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#ffffff';
              e.currentTarget.style.color = '#000000';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent';
              e.currentTarget.style.color = '#ffffff';
            }}
          >
            <Icons.X style={{ width: '16px', height: '16px' }} />
            Kapat
          </button>
        </div>
      </div>

      {/* Messages Area */}
      <div
        style={{
          flex: 1,
          overflowY: 'auto',
          padding: '32px',
          maxWidth: '1200px',
          width: '100%',
          margin: '0 auto',
          position: 'relative',
        }}
      >
        {/* Orbit Animation - Only shown when loading */}
        {loading && (
          <div
            style={{
              position: 'fixed',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              pointerEvents: 'none',
              zIndex: 1,
              overflow: 'hidden',
            }}
          >
            {CRYPTO_SYMBOLS.map((symbol, index) => {
              const delay = index * 0.3;
              const duration = 3 + (index % 3) * 0.5;
              const xOffset = (index % 3) * 30 - 30;

              return (
                <div
                  key={`${symbol}-${index}`}
                  style={{
                    position: 'absolute',
                    left: `${50 + xOffset}%`,
                    bottom: '-50px',
                    animation: `orbitUp ${duration}s ease-out ${delay}s infinite`,
                    fontSize: '24px',
                    fontWeight: 'bold',
                    color: '#00ff88',
                    textShadow: '0 0 10px #00ff88, 0 0 20px #00ff88',
                    opacity: 0,
                  }}
                >
                  {symbol}
                </div>
              );
            })}
          </div>
        )}
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '100px 20px', color: '#8b8b8b' }}>
            <Icons.Bot style={{ width: '80px', height: '80px', color: '#444', marginBottom: '24px' }} />
            <h2 className="neon-text" style={{ fontSize: '28px', marginBottom: '12px' }}>
              AI Asistanınız Hazır
            </h2>
            <p style={{ fontSize: '16px', maxWidth: '600px', margin: '0 auto' }}>
              Piyasa verileri, strateji analizi ve coin bilgileri hakkında sorularınızı sorun.
            </p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              style={{
                marginBottom: '32px',
                display: 'flex',
                gap: '16px',
                alignItems: 'flex-start',
              }}
            >
              <div
                style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '50%',
                  background: msg.role === 'user' ? '#ffffff' : '#000',
                  border: msg.role === 'user' ? 'none' : '2px solid #ffffff',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0,
                }}
              >
                {msg.role === 'user' ? (
                  <Icons.User style={{ width: '20px', height: '20px', color: '#000' }} />
                ) : (
                  <Icons.Bot style={{ width: '20px', height: '20px', color: '#ffffff' }} />
                )}
              </div>

              <div style={{ flex: 1 }}>
                <div
                  style={{
                    color: '#8b8b8b',
                    fontSize: '12px',
                    marginBottom: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}
                >
                  <span style={{ fontWeight: 600 }}>
                    {msg.role === 'user' ? 'SİZ' : 'Ailydian AI'}
                  </span>
                  <span>{msg.timestamp.toLocaleTimeString('tr-TR')}</span>
                </div>

                <div
                  style={{
                    color: '#ffffff',
                    fontSize: '15px',
                    lineHeight: '1.7',
                  }}
                >
                  {msg.role === 'assistant' ? formatMessage(msg.content) : <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>}
                </div>
              </div>
            </div>
          ))
        )}

        {loading && (
          <div style={{ marginBottom: '32px', display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
            <div
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '50%',
                background: '#000',
                border: '2px solid #ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Icons.Bot style={{ width: '20px', height: '20px', color: '#ffffff' }} />
            </div>
            <div style={{ flex: 1 }}>
              <div style={{ color: '#8b8b8b', fontSize: '15px', marginBottom: '4px' }}>
                <span className="neon-text">Düşünüyorum</span>
                <span style={{ animation: 'pulse 1.5s ease-in-out infinite' }}>...</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '13px', color: '#666' }}>
                <Icons.Clock style={{ width: '14px', height: '14px', color: '#00ff88' }} />
                <span style={{ color: '#00ff88', fontWeight: 600, fontFamily: 'monospace' }}>
                  {elapsedTime.toFixed(1)}s
                </span>
                <span>• Tüm stratejiler + Quantum + AI analiz ediliyor...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Fixed Bottom */}
      <div
        style={{
          borderTop: '1px solid #222',
          padding: '24px 32px',
          background: '#000',
        }}
      >
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', gap: '16px' }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Mesajınızı yazın... (Enter: gönder, Shift+Enter: yeni satır)"
            disabled={loading}
            rows={3}
            style={{
              flex: 1,
              background: '#0a0a0a',
              border: '1px solid #444',
              borderRadius: '12px',
              color: '#ffffff',
              padding: '16px',
              fontSize: '15px',
              resize: 'none',
              outline: 'none',
              fontFamily: 'inherit',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = '#ffffff';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = '#444';
            }}
          />

          <button
            onClick={sendMessage}
            disabled={!input.trim() || loading}
            style={{
              background: input.trim() && !loading ? '#ffffff' : '#222',
              border: 'none',
              color: input.trim() && !loading ? '#000' : '#666',
              padding: '16px 32px',
              borderRadius: '12px',
              cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
              fontSize: '15px',
              fontWeight: 700,
              transition: 'all 0.2s',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            {loading ? (
              <>
                <Icons.Loader2 style={{ width: '20px', height: '20px', animation: 'spin 1s linear infinite' }} />
                BEKLE...
              </>
            ) : (
              <>
                <Icons.Send style={{ width: '20px', height: '20px' }} />
                GÖNDER
              </>
            )}
          </button>
        </div>

        <div
          style={{
            maxWidth: '1200px',
            margin: '16px auto 0',
            textAlign: 'center',
            color: '#666',
            fontSize: '12px',
          }}
        >
          <Icons.AlertTriangle style={{ width: '12px', height: '12px', marginRight: '6px' }} />
          AI yanıtları bilgilendirme amaçlıdır. Yatırım tavsiyesi değildir.
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }

        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.3;
          }
        }

        @keyframes orbitUp {
          0% {
            transform: translateY(0) translateX(0) rotate(0deg);
            opacity: 0;
          }
          10% {
            opacity: 1;
          }
          50% {
            transform: translateY(-50vh) translateX(30px) rotate(180deg);
            opacity: 0.8;
          }
          80% {
            opacity: 0.5;
          }
          100% {
            transform: translateY(-120vh) translateX(-20px) rotate(360deg);
            opacity: 0;
          }
        }
      `}</style>
    </div>
  );
}
