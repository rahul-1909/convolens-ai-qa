# ConvoLens Production Deployment Guide

This guide provides step-by-step instructions for deploying the ConvoLens Automated Quality & Root Cause Intelligence platform across serverless and containerized enterprise environments.

---

## Production Deployment Checklist

### Vercel Serverless (Current Production Deployment)
- [x] **Runtime Environment**: Python 3.12 with `uv` package resolver.
- [x] **Timeout Configuration**: `maxDuration: 60` configured in `vercel.json` to handle multi-turn LLM judge evaluations.
- [x] **Rewrites & Path Routing**: Configured `/api/(.*)` -> `api/index.py?__path=/$1` and root rewrites for static assets.
- [x] **Zero-Trust PII Masking**: Enabled by default on all ingested transcripts before logging or storage.
- [ ] **Environment Variables**:
  - `ANTHROPIC_API_KEY`: Required for production Claude 3.5 Sonnet evaluation judge.
  - `OPENAI_API_KEY`: Fallback judge provider.
  - `DATABASE_URL`: Connection string for persistent PostgreSQL database (defaults to local SQLite `/tmp/convolens.db` in ephemeral serverless instances).
- [ ] **Monitoring & Alerting**: Configure uptime alerts and Vercel Analytics.

---

## Deploying to Vercel (CLI)

```bash
# Install Vercel CLI
npm i -g vercel

# Link and deploy to production
vercel --prod
```

### Environment Variables on Vercel
Set through Vercel Dashboard (`Settings -> Environment Variables`) or CLI:
```bash
vercel env add ANTHROPIC_API_KEY production
vercel env add OPENAI_API_KEY production
vercel env add ENVIRONMENT production
```

---

## Containerized Deployment: Docker & Docker Compose

For enterprise deployments requiring self-hosted infrastructure, data residency compliance, or persistent HDBSCAN clustering state, deploy using Docker:

### 1. Build and Run via Docker Compose
```bash
# Clone the repository
git clone https://github.com/rahul-1909/convolens-ai-qa.git
cd convolens-ai-qa

# Copy environment variables
cp .env.example .env

# Build and start services
docker-compose up --build -d
```

### 2. Verify Container Health
```bash
curl -f http://localhost:8000/health
```

---

## Cloud Deployment: AWS ECS / GCP Cloud Run

### AWS Elastic Container Service (ECS) / Fargate
1. **Container Image**: Build and push Docker image to Amazon ECR:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t convolens:latest .
   docker tag convolens:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/convolens:latest
   docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/convolens:latest
   ```
2. **Task Definition**:
   - CPU: 1024 (1 vCPU), Memory: 2048 MB.
   - Port Mapping: `8000:8000`.
   - Health check: `CMD-SHELL, curl -f http://localhost:8000/health || exit 1`.
   - Set environment variables (`ANTHROPIC_API_KEY`, `DATABASE_URL`).
3. **Application Load Balancer (ALB)**:
   - Target group routing to container port 8000.
   - Idle timeout set to 60s.

### Google Cloud Run
Deploy with a single command using Google Cloud SDK:
```bash
gcloud run deploy convolens-ai-qa \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --timeout 60s \
  --memory 2Gi \
  --set-env-vars ENVIRONMENT=production
```

---

## Security & Enterprise Governance
- **Zero PII Leakage**: PAN, Aadhaar, and phone numbers are scrubbed prior to any persistence or external LLM API calls.
- **Role-Based API Access**: Configure reverse-proxy API gateways (e.g., Kong, Cloudflare) with Bearer token authentication in front of `/evaluate/*` endpoints.
- **Fail-Safe Evaluation Fallback**: If LLM provider APIs encounter rate-limits or network interruptions, ConvoLens falls back to internal deterministic rule-based evaluation.
