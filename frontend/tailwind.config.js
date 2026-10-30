/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        lightBg: "#F8FAFC",
        lightSecondary: "#FFFFFF",
        lightCard: "#FFFFFF",
        lightBorder: "#E2E8F0",
        cyanAccent: "#0284C7",
        indigoAccent: "#4F46E5",
        successGreen: "#16A34A",
        warningYellow: "#D97706",
        errorRed: "#DC2626",
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        heading: ['Manrope', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
