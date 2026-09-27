import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Fail loudly instead of hopping to 5174: the AI service's CORS and the demo flow expect 5173.
  server: { port: 5173, strictPort: true },
})
