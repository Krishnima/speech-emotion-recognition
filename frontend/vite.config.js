import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api calls are forwarded to the Flask server, so no CORS setup is needed in dev.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:5000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
