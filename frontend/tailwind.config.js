/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        darkbg: "#0A0F1E",
        primary: "#6366F1",
        secondary: "#22D3EE",
        success: "#10B981",
        warning: "#F59E0B",
        danger: "#EF4444",
        textPrimary: "#F1F5F9",
        textMuted: "#94A3B8",
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
