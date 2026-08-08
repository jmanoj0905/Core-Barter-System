import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl(), tailwindcss()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    https: true,
    proxy: {
      '/session': { target: 'http://localhost:8000', changeOrigin: true },
      '/verdict': { target: 'http://localhost:8000', changeOrigin: true },
      '/trust':   { target: 'http://localhost:8000', changeOrigin: true },
      '/safety':  { target: 'http://localhost:8000', changeOrigin: true },
      '/window':  { target: 'http://localhost:8000', changeOrigin: true },
      '/warnings':{ target: 'http://localhost:8000', changeOrigin: true },
      '/wallet':  { target: 'http://localhost:8000', changeOrigin: true },
      '/escrow':  { target: 'http://localhost:8000', changeOrigin: true },
      '/settlement': { target: 'http://localhost:8000', changeOrigin: true },
      '/transactions': { target: 'http://localhost:8000', changeOrigin: true },
      '/users':        { target: 'http://localhost:8000', changeOrigin: true },
      '/ws':      { target: 'ws://localhost:8000',   ws: true, changeOrigin: true },
      '/audio':   { target: 'ws://localhost:8001',   ws: true, changeOrigin: true },
      '/stt':     { target: 'http://localhost:8001', changeOrigin: true },
    },
  },
})
