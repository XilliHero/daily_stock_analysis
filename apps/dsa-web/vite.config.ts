import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

const packageJson = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf-8'),
) as { version?: string }
const buildTime = new Date().toISOString()

// https://vite.dev/config/
export default defineConfig({
  define: {
    __APP_PACKAGE_VERSION__: JSON.stringify(packageJson.version ?? '0.0.0'),
    __APP_BUILD_TIME__: JSON.stringify(buildTime),
  },
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  server: {
    host: '0.0.0.0',  // 允许公网访问
    port: 5173,       // 默认端口
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // Output to the project-root static/ folder served by the backend.
    outDir: path.resolve(__dirname, '../../static'),
    emptyOutDir: true,
    // Split heavy vendors into their own chunks so the main bundle stays small.
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (!id.includes('node_modules')) return undefined
          // Isolate React CORE only, matched precisely. It is a dependency leaf,
          // so nothing forms a cross-chunk init cycle with it. Matching broadly
          // (e.g. any "/react") sweeps in "*/react-*" ecosystem packages whose
          // own deps land in `vendor`, creating a react-vendor <-> vendor cycle
          // that leaves React undefined at eval ("undefined reading 'forwardRef'").
          if (/[\\/]node_modules[\\/](react|react-dom|react-is|scheduler)[\\/]/.test(id))
            return 'react-vendor'
          if (id.includes('recharts') || id.includes('/d3-')) return 'charts'
          if (
            id.includes('react-markdown') ||
            id.includes('remark') ||
            id.includes('micromark') ||
            id.includes('mdast') ||
            id.includes('unist') ||
            id.includes('hast') ||
            id.includes('remove-markdown')
          )
            return 'markdown'
          if (id.includes('motion') || id.includes('framer')) return 'motion'
          if (id.includes('react-router')) return 'router'
          return 'vendor'
        },
      },
    },
  },
})
