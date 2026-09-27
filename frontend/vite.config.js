import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
    allowedHosts: process.env.VITE_ALLOW_ALL_HOSTS === 'true' ? true : ['localhost', '127.0.0.1'],
  },
});
