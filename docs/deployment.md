# 部署注意事项

## 真实 IP

- uvicorn 开了 `--proxy-headers`，`FORWARDED_ALLOW_IPS` 配成 nginx 容器的固定地址（`docker-compose.yml`），
  不要配 `*`。改网段或 nginx 地址时两处一起改。
- api 的 8000 端口只绑本机，不要暴露到公网。

## 端口

compose 里 api、Postgres、Redis 都只绑 `127.0.0.1`；前端端口由 `FRONTEND_PORT` 指定，默认 5173，
服务器上配 80。

## `.env`

- `DEV_LOGIN_ENABLED` 必须是 false：它是直接签发令牌的接口。compose 会读项目目录的 `.env`，
  照搬本地的 true 就对公网开放了。
- 服务器单独生成 `JWT_SECRET`，不和本地共用。

## 发布

服务器上代码在 `~/enterprise-ai-assistant`，`.env` 单独维护、不进仓库。

发布前先对比两边的 `.env`。除了 `DEV_LOGIN_ENABLED`、`JWT_SECRET`、`FRONTEND_PORT` 这三项故意不同，
其余有差异都要同步到服务器，否则代码用到的新配置在服务器上取默认值或启动失败。输出会带出密钥原文：

```bash
diff <(grep -E '^[A-Z_]+=' .env | grep -vE '^(DEV_LOGIN_ENABLED|JWT_SECRET|FRONTEND_PORT)=' | sort) \
  <(ssh ubuntu@43.198.216.245 "grep -E '^[A-Z_]+=' ~/enterprise-ai-assistant/.env | grep -vE '^(DEV_LOGIN_ENABLED|JWT_SECRET|FRONTEND_PORT)=' | sort")
```

没有输出就是一致。本地推送代码后执行：

```bash
ssh ubuntu@43.198.216.245 'cd ~/enterprise-ai-assistant && git pull && sudo docker compose up -d --build'
```
