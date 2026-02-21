'use client';

/**
 * Ailydian AI ASSISTANT
 * Advanced AI + All Strategies Combined
 * Real-time market data integration
 */

import { useState, useEffect, useRef } from 'react';
import { Icons } from './Icons';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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
    setLoading(true);

    try {
      const response = await fetch('/api/ai-assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.slice(-5), // Last 5 messages for context
        }),
      });

      const result = await response.json();

      if (result.success) {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: result.response,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error('AI Assistant error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '60px',
          height: '60px',
          borderRadius: '50%',
          background: '#000',
          border: '2px solid #ffffff',
          color: '#ffffff',
          cursor: 'pointer',
          zIndex: 999,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 20px rgba(255,255,255,0.3)',
          transition: 'all 0.3s',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = 'scale(1.1)';
          e.currentTarget.style.boxShadow = '0 6px 30px rgba(255,255,255,0.5)';
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = 'scale(1)';
          e.currentTarget.style.boxShadow = '0 4px 20px rgba(255,255,255,0.3)';
        }}
      >
        {isOpen ? (
          <Icons.X style={{ width: '28px', height: '28px' }} />
        ) : (
          <Icons.Bot style={{ width: '32px', height: '32px' }} />
        )}
      </button>

      {/* Chat Modal */}
      {isOpen && (
        <div
          className="ai-assistant-modal"
          style={{
            position: 'fixed',
            bottom: '100px',
            right: '24px',
            width: '420px',
            height: '600px',
            background: '#000',
            border: '2px solid #ffffff',
            borderRadius: '16px',
            zIndex: 998,
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 8px 40px rgba(255,255,255,0.2)',
          }}
        >
          {/* Header */}
          <div style={{ padding: '20px', borderBottom: '1px solid #444', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Icons.Bot style={{ width: '24px', height: '24px', color: '#ffffff' }} />
            <div style={{ flex: 1 }}>
              <h3 className="neon-text" style={{ fontSize: '18px', margin: 0 }}>Ailydian AI Asistan</h3>
              <p style={{ color: '#666', fontSize: '12px', margin: 0 }}>Gelişmiş AI + Tüm Stratejiler</p>
            </div>
            <button
              onClick={() => setMessages([])}
              style={{
                background: 'transparent',
                border: '1px solid #444',
                color: '#666',
                padding: '6px 12px',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px',
              }}
            >
              TEMİZLE
            </button>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {messages.length === 0 && (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: '#666' }}>
                <Icons.Bot style={{ width: '48px', height: '48px', color: '#444', margin: '0 auto 16px' }} />
                <p>Merhaba! Size nasıl yardımcı olabilirim?</p>
                <div style={{ fontSize: '12px', marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  <div style={{ color: '#888' }}>Örnek sorular:</div>
                  <div style={{ color: '#666' }}>• BTC için şu anki stratejiler nedir?</div>
                  <div style={{ color: '#666' }}>• En yüksek hacimli koinler hangileri?</div>
                  <div style={{ color: '#666' }}>• ETH için AI analizi yap</div>
                </div>
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                style={{
                  alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  maxWidth: '80%',
                }}
              >
                <div
                  style={{
                    background: msg.role === 'user' ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.05)',
                    border: `1px solid ${msg.role === 'user' ? '#ffffff' : '#444'}`,
                    borderRadius: '12px',
                    padding: '12px',
                    color: '#fff',
                    fontSize: '14px',
                    lineHeight: '1.5',
                  }}
                >
                  {msg.content}
                </div>
                <div style={{ fontSize: '10px', color: '#666', marginTop: '4px', textAlign: msg.role === 'user' ? 'right' : 'left' }}>
                  {msg.timestamp.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ alignSelf: 'flex-start', maxWidth: '80%' }}>
                <div
                  style={{
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid #444',
                    borderRadius: '12px',
                    padding: '12px',
                    color: '#fff',
                    fontSize: '14px',
                    display: 'flex',
                    gap: '8px',
                    alignItems: 'center',
                  }}
                >
                  <div className="loading-dots">
                    <span>.</span><span>.</span><span>.</span>
                  </div>
                  Düşünüyorum
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div style={{ padding: '16px', borderTop: '1px solid #444' }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Bir şey sorun..."
                disabled={loading}
                rows={2}
                style={{
                  flex: 1,
                  background: '#000',
                  border: '1px solid #444',
                  borderRadius: '8px',
                  padding: '12px',
                  color: '#fff',
                  fontSize: '14px',
                  outline: 'none',
                  resize: 'none',
                  fontFamily: 'inherit',
                }}
              />
              <button
                onClick={sendMessage}
                disabled={!input.trim() || loading}
                style={{
                  background: input.trim() && !loading ? '#ffffff' : 'transparent',
                  border: '1px solid #ffffff',
                  color: input.trim() && !loading ? '#000' : '#666',
                  padding: '12px 20px',
                  borderRadius: '8px',
                  cursor: input.trim() && !loading ? 'pointer' : 'not-allowed',
                  fontSize: '14px',
                  fontWeight: '600',
                  transition: 'all 0.2s',
                }}
              >
                GÖNDER
              </button>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        @keyframes dots {
          0%, 20% { opacity: 0.2; }
          50% { opacity: 1; }
          100% { opacity: 0.2; }
        }

        .loading-dots span {
          animation: dots 1.4s infinite;
          display: inline-block;
        }

        .loading-dots span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .loading-dots span:nth-child(3) {
          animation-delay: 0.4s;
        }

        /* Mobile Responsive - AI Assistant */
        @media (max-width: 768px) {
          .ai-assistant-modal {
            width: calc(100vw - 32px) !important;
            right: 16px !important;
            left: 16px !important;
            height: calc(100vh - 140px) !important;
            bottom: 80px !important;
          }
        }

        @media (max-width: 480px) {
          .ai-assistant-modal {
            width: calc(100vw - 16px) !important;
            right: 8px !important;
            left: 8px !important;
            height: calc(100vh - 120px) !important;
            bottom: 80px !important;
            border-radius: 12px !important;
          }
        }
      `}</style>
    </>
  );
}
