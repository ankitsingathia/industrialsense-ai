import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs separately on 8765; the built app is served by FastAPI itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
