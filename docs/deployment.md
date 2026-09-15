# 部署注意事项

## 真实 IP

`IP_TOKEN_BUDGET` 按连接对端地址计数。隔了代理取不到真实 IP，所有用户会共用一份额度。

- nginx 传 `proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;`（`frontend/nginx.conf` 现在没传）。
- uvicorn 加 `--proxy-headers`，`FORWARDED_ALLOW_IPS` 配代理地址，不要配 `*`（`Dockerfile` 现在没加）。
- 不要把 api 的 8000 端口暴露到公网。
