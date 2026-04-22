import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
    globals: true,
  },
  server: {
    proxy: {
      '/ask': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/series': 'http://localhost:8000',
    },
  },
})
