import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:4000',
        changeOrigin: true,
        // Increase body size limit for large base64 images
        timeout: 60000,
        // Vite uses http-proxy-middleware which handles large bodies by default
        // but we can be explicit about it
      }
    }
  }
});

