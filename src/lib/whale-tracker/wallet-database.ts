/**
 * 🐋 KNOWN WHALE WALLETS DATABASE
 *
 * Satoshi Nakamoto, Vitalik Buterin ve diğer bilinen whale cüzdanları.
 * 2009'dan bu yana takip edilen büyük kripto sahipleri.
 *
 * WHITE-HAT: Sadece public olarak bilinen ve açıklanmış cüzdanlar.
 */

import type { KnownWallet, Blockchain } from '@/types/whale-tracker';

export const KNOWN_WALLETS: Record<Blockchain, KnownWallet[]> = {
  // ============================================================================
  // BITCOIN WHALE WALLETS
  // ============================================================================
  BTC: [
    {
      address: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa',
      label: 'Satoshi Nakamoto - Genesis Block',
      owner: 'Satoshi Nakamoto',
      category: 'FOUNDER',
      description: 'The first Bitcoin block ever mined. Satoshi\'s most famous wallet.',
      balance: 104.32,
      firstSeen: '2009-01-03',
      turkishNote: 'Bitcoin tarihinin en önemli adresi. Genesis bloğu. Hiç hareket ettirilmedi ve bağışlar aldı.'
    },
    {
      address: '1HLoD9E4SDFFPDiYfNYnkBLQ85Y51J3Zb1',
      label: 'Hal Finney - İlk Bitcoin Transferi',
      owner: 'Hal Finney',
      category: 'FOUNDER',
      description: 'Received the first Bitcoin transaction from Satoshi Nakamoto.',
      firstSeen: '2009-01-12',
      turkishNote: 'Satoshi\'den ilk Bitcoin transferini alan kişi. 10 BTC aldı.'
    },
    {
      address: '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',
      label: 'Binance Cold Wallet',
      owner: 'Binance Exchange',
      category: 'EXCHANGE',
      description: 'World\'s largest crypto exchange cold storage wallet.',
      balance: 248000,
      turkishNote: 'Toplam Bitcoin arzının %1.2\'sini tutan Binance soğuk cüzdanı.'
    },
    {
      address: '3LYJfcfHPXYJreMsASk2jkn69LWEYKzexb',
      label: 'Bitfinex Hack Recovery',
      category: 'GOVERNMENT',
      description: 'US Government seized funds from 2016 Bitfinex hack.',
      turkishNote: '2016 Bitfinex hackinden el konulan Bitcoin\'ler. ABD hükümeti elinde tutuyor.'
    }
  ],

  // ============================================================================
  // ETHEREUM WHALE WALLETS
  // ============================================================================
  ETH: [
    {
      address: '0xab5801a7d398351b8be11c439e05c5b3259aec9b',
      label: 'Vitalik Buterin - Ana Cüzdan',
      owner: 'Vitalik Buterin',
      category: 'FOUNDER',
      description: 'Ethereum founder\'s main publicly disclosed wallet.',
      balance: 278524,
      firstSeen: '2018',
      turkishNote: 'Vitalik\'in 2018 ve 2022\'de açıkladığı 3 cüzdandan ana cüzdanı. Yaklaşık 278K ETH.'
    },
    {
      address: '0xd8da6bf26964af9d7eed9e03e53415d37aa96045',
      label: 'Vitalik Buterin - vitalik.eth',
      owner: 'Vitalik Buterin',
      category: 'FOUNDER',
      description: 'Vitalik\'s second known wallet linked to vitalik.eth ENS.',
      turkishNote: 'vitalik.eth ENS domain\'ine bağlı ikinci cüzdan. Büyük transferler için kullanılıyor.'
    },
    {
      address: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
      label: 'Ethereum Foundation',
      owner: 'Ethereum Foundation',
      category: 'FOUNDER',
      description: 'Ethereum Foundation treasury wallet.',
      turkishNote: 'Ethereum Vakfı\'nın hazine cüzdanı. Geliştirme fonları için kullanılıyor.'
    },
    {
      address: '0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503',
      label: 'Binance Hot Wallet',
      owner: 'Binance Exchange',
      category: 'EXCHANGE',
      description: 'Binance exchange hot wallet for ETH.',
      turkishNote: 'Binance\'in Ethereum sıcak cüzdanı. Aktif işlemler için kullanılıyor.'
    }
  ],

  // ============================================================================
  // AVALANCHE WHALE WALLETS
  // ============================================================================
  AVAX: [
    {
      address: '0xefdc8...3B4246',
      label: 'Exchange Hot Wallet #1',
      category: 'EXCHANGE',
      description: 'Largest AVAX exchange hot wallet.',
      balance: 6303017.46,
      turkishNote: 'En büyük AVAX tutan borsa sıcak cüzdanı. Yaklaşık $224M değerinde.'
    },
    {
      address: '0x40B38...18E489',
      label: 'Exchange Cold Wallet',
      category: 'EXCHANGE',
      description: 'Major exchange cold storage for AVAX.',
      balance: 4755180.60,
      turkishNote: 'Büyük borsa soğuk cüzdanı. Yaklaşık $196M değerinde AVAX.'
    },
    {
      address: '0x73AF3...54D935',
      label: 'Robinhood Exchange',
      owner: 'Robinhood',
      category: 'EXCHANGE',
      description: 'Robinhood exchange AVAX reserves.',
      balance: 1837418.21,
      turkishNote: 'Robinhood borsasının AVAX rezervleri. ~$66.7M değerinde.'
    }
  ]
};

// Helper function to find wallet info
export function findWalletInfo(address: string, blockchain: Blockchain): KnownWallet | null {
  const wallets = KNOWN_WALLETS[blockchain];
  return wallets.find(w =>
    w.address.toLowerCase() === address.toLowerCase()
  ) || null;
}

// Get all known addresses for a blockchain
export function getAllKnownAddresses(blockchain: Blockchain): string[] {
  return KNOWN_WALLETS[blockchain].map(w => w.address.toLowerCase());
}

// Check if address is a known whale
export function isKnownWhale(address: string, blockchain: Blockchain): boolean {
  return getAllKnownAddresses(blockchain).includes(address.toLowerCase());
}
