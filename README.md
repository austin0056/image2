# image2 — 简易生图工具

基于 `gpt-image-2`（haochi.moon9.cloud 中转）的最小可用生图站点。

- 用户面板：用 access key 登录 → 生图 / 矢量图标 / 文字转 CAD → 下载 / 历史
- 管理面板：密码登录 → 建用户 / 充值 / 看记录
- 计费：每次生成按对应单价扣费，失败自动退款

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
   - `CLAUDE_KEY`（图标快速模式 + 文字转 CAD 用；可选 `PRICE_CAD_CENTS`）
4. 绑定域名（Zeabur 默认子域自动 HTTPS）。
5. 首次访问 `/admin` 登录后建用户、充值。

## 文字转 CAD（build123d）

用户面板「文字转 CAD」标签：输入自然语言需求 → AI（Claude 中转）编写 build123d
Python 代码 → **在服务器子进程里执行** → 产出 STEP / STL / GLB，前端用
`<model-viewer>` 做浏览器端 3D 预览，按 `PRICE_CAD_CENTS` 计费、失败自动退款。

相关环境变量：

- `CLAUDE_KEY` / `CLAUDE_BASE` / `CLAUDE_MODEL`：与图标「快速模式」复用同一套
- `PRICE_CAD_CENTS`：单次价格，默认 `1000`（¥10，因生成慢、算力重）
- `CAD_EXEC_TIMEOUT`：子进程执行超时秒数，默认 `60`

**部署注意（重要）：**

- `requirements.txt` 里的 `build123d` 会拉入 `cadquery-ocp`（OpenCascade 原生 wheel，
  体积数百 MB），显著增大镜像与冷启动时间，确认平台磁盘/构建超时充足。
- Linux 容器通常还需系统库 `libgl1`、`libglu1-mesa`、`libxrender1`、`libxext6`。
  Dockerfile/Nixpacks 里 `apt-get install` 补齐；上线前务必验证
  `python -c "import build123d"` 能成功。
- **安全**：CAD 功能会执行模型生成的 Python 代码。当前用「子进程 + 超时 +
  清空环境变量 + 临时目录 + 危险调用静态拦截」做务实隔离（见 `app/cad_runner.py`），
  这是降低风险而非根除。生产硬化的理想方案是独立容器 / gVisor 沙箱（后续 TODO）。

## 安全提示

- `.env` 已加入 `.gitignore`，不要提交真 key
- `ADMIN_PASSWORD` 部署后立即改强密码
- 文件不直接暴露给浏览器，统一通过应用层鉴权后流式转发
- CAD 代码执行隔离见上节「文字转 CAD」的安全说明
