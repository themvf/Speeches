import type { Config } from "tailwindcss";
import colors from "tailwindcss/colors.js";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "var(--ink)",
        mist: "var(--mist)",
        dune: "var(--dune)",
        // A bare string here REPLACES Tailwind's amber scale rather than
        // extending it, which silently killed every `amber-<number>` utility in
        // the app - 18 files styling warnings that then rendered in the ordinary
        // ink colour. Keeping the scale and putting the brand token on DEFAULT
        // means `text-amber` still resolves to var(--amber) and `amber-300`
        // works again.
        amber: { ...colors.amber, DEFAULT: "var(--amber)" },
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