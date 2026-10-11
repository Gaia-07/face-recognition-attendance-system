# Production Deployment & Operations Guide

## 1. Overview & Architecture

FRAS is packaged for containerized deployment with microservice separation between the REST API (FastAPI) and the presentation layer (Streamlit).

```
                     ┌───────────────────────────┐
                     │   NGINX / REVERSE PROXY   │
                     │  (SSL/TLS Termination)    │
                     └──────┬─────────────┬──────┘
                            │             │
          ┌─────────────────┘             └─────────────────┐
          │ :8000                                           │ :8501
          ▼                                                 ▼
┌───────────────────────────┐                     ┌───────────────────────────┐
│       FASTAPI API         │                     │    STREAMLIT PORTAL       │
│  (Uvicorn ASGI Workers)   │                     │  (Websocket UI Dashboard) │
└─────────────┬─────────────┘                     └─────────────┬─────────────┘
              │                                                 │
              └─────────────────┐             ┌─────────────────┘
                                ▼             ▼
                         ┌───────────────────────────┐
                         │   PERSISTENT VOLUME       │
                         │ • data/fras.db (SQLite)   │
                         │ • models/avatar_cnn.onnx  │
                         │ • logs/audit.jsonl        │
                         └───────────────────────────┘
```

---

## 2. Docker & Container Orchestration

### 2.1 Single-Command Launch via Docker Compose

```bash
# Build and run containers in detached mode
docker compose up -d --build

# Inspect running services
docker compose ps

# View unified real-time logs
docker compose logs -f
```

### 2.2 Docker Service Topology
- **`api` service**:
  - Image: `fras-app:latest`
  - Command: `uvicorn fras.api.main:app --host 0.0.0.0 --port 8000`
  - Healthcheck: polls `http://localhost:8000/healthz` every 10s.
  - Exposed Port: `8000:8000`
- **`dashboard` service**:
  - Image: `fras-app:latest`
  - Command: `streamlit run src/fras/app/Home.py --server.port 8501 --server.address 0.0.0.0`
  - Exposed Port: `8501:8501`

### 2.3 Vercel Serverless Deployment (One-Click)

FRAS includes native Vercel serverless integration (`vercel.json`, `api/index.py`, and `requirements.txt`):

1. **Connect Repository on Vercel**:
   - Navigate to [https://vercel.com/new](https://vercel.com/new).
   - Select and import your GitHub repository: `Gaia-07/face-recognition-attendance-system`.
   - Leave the Root Directory as `./` and click **Deploy**.
2. **Architecture on Vercel**:
   - Vercel automatically deploys the ASGI FastAPI backend via `@vercel/python` using `api/index.py`.
   - The ONNX model (`models/avatar_cnn.onnx`) is executed on CPU via ONNX Runtime without requiring PyTorch wheels, keeping bundle size well under Vercel's 250 MB ceiling.
   - The root URL (`/`) serves an interactive, mobile-responsive Tailwind CSS biometric check-in kiosk.
   - Swagger documentation is available at `/docs`, and `/healthz` provides system diagnostics.


---

## 3. Environment Variables & Secrets Management

All production configurations are injected via environment variables (see `.env.example`):

| Variable | Description | Production Default / Example |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Deployment environment | `production` |
| `SECRET_KEY` | JWT secret signature key | 32+ character random hex string |
| `BIOMETRIC_ENCRYPTION_KEY` | Fernet 32-byte url-safe base64 key | Generate via `cryptography.fernet.Fernet.generate_key()` |
| `DATABASE_URL` | SQLAlchemy connection URI | `sqlite:///data/fras.db` (or PostgreSQL URI) |
| `BACKEND_NAME` | Embedding engine | `avatar_cnn` (demo) or `sface` (webcam) |
| `CALIBRATED_THRESHOLD` | Biometric decision threshold ($\tau^*$) | `0.653` |
| `LATE_CUTOFF_TIME` | Official late check-in cutoff | `09:15:00` |
| `AUDIT_LOG_PATH` | Path to append-only audit trail | `logs/audit.jsonl` |

### Key Generation Command
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## 4. Production Nginx Reverse Proxy Configuration

Streamlit requires WebSocket support (`Upgrade` header), while FastAPI serves standard HTTP:

```nginx
# /etc/nginx/conf.d/fras.conf

upstream fastapi_backend {
    server 127.0.0.1:8000;
}

upstream streamlit_portal {
    server 127.0.0.1:8501;
}

server {
    listen 80;
    server_name fras.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name fras.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/fras.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fras.yourdomain.com/privkey.pem;

    # FastAPI REST endpoints
    location /api/ {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /healthz {
        proxy_pass http://fastapi_backend/healthz;
    }

    # Streamlit Dashboard (with WebSocket support)
    location / {
        proxy_pass http://streamlit_portal;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

---

## 5. Security & Hardening Checklist

- [x] **Non-Root Execution**: Container runs under dedicated non-root user `appuser` (UID 1000).
- [x] **Zero Photographic Persistence**: Images processed in-memory only.
- [x] **Biometric Encryption**: SQLite database contains zero plaintext embeddings.
- [x] **Strict Secret Hygiene**: `.env` and SQLite `.db` are git-ignored.
- [x] **Readiness & Liveness Probes**: Implemented at `/healthz`.
- [x] **Audit Log Integrity**: Append-only file permissions configured for `logs/audit.jsonl`.
