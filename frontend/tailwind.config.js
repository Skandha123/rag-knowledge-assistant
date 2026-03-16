/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['"DM Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['"Syne"', 'sans-serif'],
      },
      colors: {
        surface: {
          50: '#f8f7f4',
          100: '#f0ede7',
          200: '#e0dbd2',
          900: '#1a1916',
          950: '#100f0d',
        },
        accent: {
          DEFAULT: '#e8602c',
          light: '#f0855a',
          dark: '#c94d20',
        },
        ink: {
          DEFAULT: '#1a1916',
          muted: '#6b6760',
          faint: '#9c9892',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1)',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
        'typing': 'typing 1.2s steps(3, end) infinite',
      },
      keyframes: {
        fadeIn: { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(12px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        pulseSoft: { '0%, 100%': { opacity: 1 }, '50%': { opacity: 0.5 } },
        typing: { '0%': { content: '.' }, '33%': { content: '..' }, '66%': { content: '...' } },
      },
    },
  },
  plugins: [],
}
