import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // 显式使用 IPv4 地址，因为 `localhost` 可能解析为 ::1，
  // 而本地启动的 Uvicorn 进程可能只监听 127.0.0.1。
  server: {proxy: {"/api": "http://127.0.0.1:8000"}},
});
