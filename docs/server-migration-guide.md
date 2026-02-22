# 云主机迁移与 HTTPS 配置指南

本文档适用于将后端服务迁移到新云主机，并配置 HTTPS 证书。

- **域名**: `bce.kkmsee.com`
- **镜像仓库**: 腾讯云 TCR (`ccr.ccs.tencentyun.com/mz_personal_namespace/golf_backend`)
- **后端端口**: 8000
- **项目目录**: `/opt/golf-backend`

---

## 一、新主机初始化

以下命令假设新主机为 Ubuntu/Debian 系统。

### 1.1 安装 Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo systemctl enable --now docker
```

### 1.2 安装 Docker Compose

```bash
sudo apt update && sudo apt install -y docker-compose-plugin
```

### 1.3 创建项目目录和数据目录

```bash
sudo mkdir -p /opt/golf-backend/data/{raw_videos,processed_videos,metadata,pose_data,analysis_results,visualizations}
```

### 1.4 创建 docker-compose.yml

```bash
sudo tee /opt/golf-backend/docker-compose.yml << 'EOF'
services:
  backend:
    image: ccr.ccs.tencentyun.com/mz_personal_namespace/golf_backend:latest
    container_name: golf-backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - TZ=Asia/Shanghai
EOF
```

### 1.5 登录 TCR 并启动服务

```bash
sudo docker login ccr.ccs.tencentyun.com -u <TCR用户名> -p <TCR密码 tencent1>
cd /opt/golf-backend && sudo docker compose up -d
```

---

## 二、Nginx 反向代理 + HTTPS 证书

### 2.1 安装 Nginx 和 Certbot

```bash
sudo apt install -y nginx certbot python3-certbot-nginx
```

### 2.2 配置 Nginx 反向代理

```bash
sudo tee /etc/nginx/sites-available/golf-api << 'EOF'
server {
    listen 80;
    server_name bce.kkmsee.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 100M;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/golf-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### 2.3 申请 HTTPS 证书

```bash
sudo certbot --nginx -d bce.kkmsee.com
```

Certbot 会自动修改 Nginx 配置添加 SSL，并设置自动续期。

---

## 三、DNS 解析

在域名管理后台设置 A 记录：

| 主机记录 | 记录类型 | 记录值 |
|---------|---------|--------|
| bce     | A       | 新主机 IP |

---

## 四、GitHub Secrets

在 GitHub 仓库 → Settings → Secrets and variables → Actions 中更新以下 3 个 Secret：

| Secret | 说明 |
|--------|------|
| `CVM_HOST` | 新主机 IP |
| `CVM_USERNAME` | SSH 用户名 |
| `CVM_SSH_KEY` | 新主机的 SSH 私钥 |

> `TCR_USERNAME` 和 `TCR_PASSWORD` 与主机无关，无需修改。

---

## 五、数据迁移（可选）

如旧主机上有需要保留的视频和分析数据：

```bash
scp -r old-server:/opt/golf-backend/data/ /opt/golf-backend/data/
```

---

## 六、微信公众平台

在 **微信公众平台 → 开发管理 → 服务器域名** 中确认 request 合法域名包含：

```
https://bce.kkmsee.com
```

> 域名未变更时无需修改。

---

## 七、无需修改的部分

- 代码仓库和 CI/CD 流水线逻辑
- TCR 镜像仓库配置
- `utils/config.js` 中的域名配置
- 微信公众平台域名配置（域名未变更时）
