import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        mist: "var(--mist)",
        dune: "var(--dune)",
        amber: "var(--amber)",
        steel: "var(--steel)"
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.5rem"
      },
      boxShadow: {
        panel: "0 14px 40px rgba(15, 34, 56, 0.12)"
      }
    }
  },
  plugins: []
};

export default config;