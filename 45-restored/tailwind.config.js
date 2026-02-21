/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'accent-1': '#FF214D',
        'accent-2': '#FF6A45',
        'dark-bg': '#0A0A0B',
        'light-text': '#F3F4F6',
        'gain': '#10b981',
        'loss': '#ef4444',
        'neutral': '#6b7280',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'gradient-neon': 'linear-gradient(135deg, #FF214D 0%, #FF6A45 100%)',
      },
    },
  },
  plugins: [],
}

