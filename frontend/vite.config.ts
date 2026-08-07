import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Use an explicit IPv4 address because `localhost` may resolve to ::1 while
  // a locally started Uvicorn process is listening only on 127.0.0.1.
  server: {proxy: {"/api": "http://127.0.0.1:8000"}},
});
