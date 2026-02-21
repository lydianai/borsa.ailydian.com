/**
 * 🔊 AUDIO NOTIFICATION SERVICE
 *
 * Cross-strategy audio notification system with:
 * - Multiple sound types (signal, alert, warning, success)
 * - Volume control
 * - Enable/disable toggle
 * - Browser audio policy compliance
 * - White-hat: User consent required
 */

export type SoundType = 'signal' | 'alert' | 'warning' | 'success' | 'info';

interface AudioConfig {
  enabled: boolean;
  volume: number; // 0-1
  soundFiles: Record<SoundType, string>;
}

class AudioNotificationService {
  private config: AudioConfig;
  private audioContext: AudioContext | null = null;
  private audioBuffers: Map<SoundType, AudioBuffer> = new Map();

  constructor() {
    this.config = {
      enabled: this.getStoredEnabled(),
      volume: this.getStoredVolume(),
      soundFiles: {
        signal: '/notification-sound.mp3',
        alert: '/notification-sound.mp3',
        warning: '/notification-sound.mp3',
        success: '/notification-sound.mp3',
        info: '/notification-sound.mp3',
      },
    };

    // Initialize audio context on user interaction
    if (typeof window !== 'undefined') {
      document.addEventListener('click', this.initAudioContext.bind(this), { once: true });
    }
  }

  /**
   * Initialize Web Audio API context
   */
  private initAudioContext() {
    if (!this.audioContext && typeof window !== 'undefined') {
      try {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        console.log('[Audio Service] AudioContext initialized');
      } catch (error) {
        console.error('[Audio Service] Failed to initialize AudioContext:', error);
      }
    }
  }

  /**
   * Get stored enabled state from localStorage
   */
  private getStoredEnabled(): boolean {
    if (typeof window === 'undefined') return false;
    const stored = localStorage.getItem('audioNotifications');
    return stored ? JSON.parse(stored) : false;
  }

  /**
   * Get stored volume from localStorage
   */
  private getStoredVolume(): number {
    if (typeof window === 'undefined') return 0.7;
    const stored = localStorage.getItem('audioVolume');
    return stored ? parseFloat(stored) : 0.7;
  }

  /**
   * Enable audio notifications
   */
  enable() {
    this.config.enabled = true;
    if (typeof window !== 'undefined') {
      localStorage.setItem('audioNotifications', 'true');
    }
    this.initAudioContext();
    console.log('[Audio Service] Enabled');
  }

  /**
   * Disable audio notifications
   */
  disable() {
    this.config.enabled = false;
    if (typeof window !== 'undefined') {
      localStorage.setItem('audioNotifications', 'false');
    }
    console.log('[Audio Service] Disabled');
  }

  /**
   * Toggle audio notifications
   */
  toggle(): boolean {
    if (this.config.enabled) {
      this.disable();
    } else {
      this.enable();
    }
    return this.config.enabled;
  }

  /**
   * Check if audio is enabled
   */
  isEnabled(): boolean {
    return this.config.enabled;
  }

  /**
   * Set volume (0-1)
   */
  setVolume(volume: number) {
    this.config.volume = Math.max(0, Math.min(1, volume));
    if (typeof window !== 'undefined') {
      localStorage.setItem('audioVolume', this.config.volume.toString());
    }
    console.log('[Audio Service] Volume set to', this.config.volume);
  }

  /**
   * Get current volume
   */
  getVolume(): number {
    return this.config.volume;
  }

  /**
   * Play notification sound using HTML5 Audio
   * Fallback method for better browser compatibility
   */
  async play(type: SoundType = 'signal'): Promise<void> {
    if (!this.config.enabled) {
      console.log('[Audio Service] Notifications disabled, skipping sound');
      return;
    }

    try {
      const soundFile = this.config.soundFiles[type] || this.config.soundFiles.signal;
      const audio = new Audio(soundFile);
      audio.volume = this.config.volume;

      await audio.play();
      console.log(`[Audio Service] Playing sound: ${type}`);
    } catch (error: any) {
      console.error('[Audio Service] Failed to play sound:', error.message);

      // Show user-friendly message if autoplay is blocked
      if (error.name === 'NotAllowedError') {
        console.warn('[Audio Service] Browser blocked autoplay. User interaction required.');
      }
    }
  }

  /**
   * Play notification sound using Web Audio API (advanced)
   * Better performance for multiple rapid notifications
   */
  async playWebAudio(type: SoundType = 'signal'): Promise<void> {
    if (!this.config.enabled) return;

    if (!this.audioContext) {
      // Fallback to HTML5 audio if AudioContext not available
      return this.play(type);
    }

    try {
      // Resume audio context if suspended (browser autoplay policy)
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Load buffer if not cached
      if (!this.audioBuffers.has(type)) {
        const soundFile = this.config.soundFiles[type] || this.config.soundFiles.signal;
        const response = await fetch(soundFile);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
        this.audioBuffers.set(type, audioBuffer);
      }

      // Play the sound
      const source = this.audioContext.createBufferSource();
      const gainNode = this.audioContext.createGain();

      source.buffer = this.audioBuffers.get(type)!;
      gainNode.gain.value = this.config.volume;

      source.connect(gainNode);
      gainNode.connect(this.audioContext.destination);

      source.start(0);
      console.log(`[Audio Service] Playing sound (Web Audio): ${type}`);
    } catch (error: any) {
      console.error('[Audio Service] Failed to play Web Audio:', error);
      // Fallback to HTML5 audio
      return this.play(type);
    }
  }

  /**
   * Play signal notification (for trading signals)
   */
  playSignal() {
    return this.play('signal');
  }

  /**
   * Play alert notification (for urgent alerts)
   */
  playAlert() {
    return this.play('alert');
  }

  /**
   * Play warning notification
   */
  playWarning() {
    return this.play('warning');
  }

  /**
   * Play success notification
   */
  playSuccess() {
    return this.play('success');
  }

  /**
   * Play info notification
   */
  playInfo() {
    return this.play('info');
  }

  /**
   * Test audio playback
   */
  async test(): Promise<boolean> {
    try {
      await this.play('info');
      return true;
    } catch (error) {
      console.error('[Audio Service] Test failed:', error);
      return false;
    }
  }

  /**
   * Get service status
   */
  getStatus() {
    return {
      enabled: this.config.enabled,
      volume: this.config.volume,
      audioContextState: this.audioContext?.state || 'not initialized',
      bufferedSounds: Array.from(this.audioBuffers.keys()),
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup() {
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
      this.audioBuffers.clear();
      console.log('[Audio Service] Cleanup complete');
    }
  }
}

// Singleton instance
export const audioNotificationService = new AudioNotificationService();

// Export for convenience
export const playSignalSound = () => audioNotificationService.playSignal();
export const playAlertSound = () => audioNotificationService.playAlert();
export const playWarningSound = () => audioNotificationService.playWarning();
export const playSuccessSound = () => audioNotificationService.playSuccess();
export const toggleAudioNotifications = () => audioNotificationService.toggle();
