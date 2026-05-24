# image2 — 简易生图工具

基于 `gpt-image-2`（haochi.moon9.cloud 中转）的最小可用生图站点。

- 用户面板：用 access key 登录 → 输入 prompt + 可选参考图 → 生成 → 下载 / 历史
- 管理面板：密码登录 → 建用户 / 充值 / 看记录
- 计费：每次生成扣 5 分钱，失败自动退款

## 技术栈

FastAPI · PostgreSQL · MinIO（S3 兼容）· 原生 HTML/JS 前端

## 本地开发

```bash
# 1. 起依赖（PG + MinIO）
docker run -d --name pg -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# 2. 装依赖
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 3. 配置
copy .env.example .env    # 编辑 .env，至少填 UPSTREAM_KEY 和 ADMIN_PASSWORD

# 4. 启动
uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/admin> 用 `ADMIN_PASSWORD` 登录 → 新建用户 → 充值 → 复制 access key → 用户面板登录生图。

## 部署到 Zeabur

1. 把代码推到 GitHub。
2. Zeabur → New Project → 加 PostgreSQL 模板、加 MinIO 模板、加 Git 服务（选这个仓库）。
3. 在 Git 服务的 Variables 里填：
   - `UPSTREAM_KEY`
   - `ADMIN_PASSWORD`
   - `SESSION_SECRET`（任意 32 字节随机字符串）
   - `DATABASE_URL`（从 PG 模板复制）
   - `S3_ENDPOINT`（MinIO 内网，如 `http://minio.zeabur.internal:9000`）
   - `S3_ACCESS_KEY` / `S3_SECRET_KEY`（MinIO 模板里）
4. 绑定域名（Zeabur 默认子域自动 HTTPS）。
5. 首次访问 `/admin` 登录后建用户、充值。

## 安全提示

- `.env` 已加入 `.gitignore`，不要提交真 key
- `ADMIN_PASSWORD` 部署后立即改强密码
- 文件不直接暴露给浏览器，统一通过应用层鉴权后流式转发
