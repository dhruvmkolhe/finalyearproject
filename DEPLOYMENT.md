# PredictIQ Deployment Guide

Complete guide for deploying PredictIQ to production environments.

---

## 📋 Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Platform Deployment](#cloud-platform-deployment)
4. [SSL/TLS Configuration](#ssltls-configuration)
5. [Database Migration](#database-migration)
6. [Monitoring Setup](#monitoring-setup)
7. [Backup & Recovery](#backup--recovery)
8. [Troubleshooting](#troubleshooting)

---

## 🔍 Pre-Deployment Checklist

### Security
- [ ] Change all default passwords in `.env`
- [ ] Generate secure `SECRET_KEY` and `JWT_SECRET`
- [ ] Configure CORS for production domain only
- [ ] Set up SSL certificates (Let's Encrypt recommended)
- [ ] Review and restrict exposed ports
- [ ] Enable firewall rules (allow 80, 443 only)
- [ ] Set up VPC/security groups (cloud deployments)

### Configuration
- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure production database (PostgreSQL recommended)
- [ ] Set up Redis for caching
- [ ] Configure production logging level
- [ ] Set appropriate rate limits
- [ ] Configure backup schedules

### Resources
- [ ] Verify server specifications (minimum 4GB RAM, 2 CPU cores)
- [ ] Ensure sufficient disk space (20GB+ recommended)
- [ ] Set up CDN for static assets (optional)
- [ ] Configure auto-scaling policies (cloud platforms)

---

## 🐳 Docker Deployment

### Step 1: Server Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2: Clone Repository

```bash
git clone https://github.com/yourusername/predictiq.git
cd predictiq
```

### Step 3: Configure Environment

```bash
# Copy and edit environment file
cp .env.example .env
nano .env

# Required changes:
# - GROQ_API_KEY=your_actual_api_key
# - DB_PASSWORD=secure_random_password
# - REDIS_PASSWORD=secure_random_password
# - GRAFANA_PASSWORD=secure_admin_password
# - DOMAIN=yourdomain.com
# - CORS_ORIGINS=https://yourdomain.com
```

### Step 4: Launch Stack

```bash
# Pull latest images
docker-compose pull

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f backend
```

### Step 5: Initialize Database

```bash
# Run database migrations
docker-compose exec backend alembic upgrade head

# Run initial data pipeline (if needed)
docker-compose exec backend python pipeline/preprocessing.py
docker-compose exec backend python pipeline/rfm_features.py
docker-compose exec backend python pipeline/segmentation.py
docker-compose exec backend python pipeline/model_training.py
```

### Step 6: Verify Deployment

```bash
# Health check
curl http://localhost:8000/api/health

# Expected response:
# {"success": true, "data": {"status": "healthy", "models_loaded": {...}}}
```

---

## ☁️ Cloud Platform Deployment

### AWS (ECS/Fargate)

#### 1. Create ECR Repositories

```bash
aws ecr create-repository --repository-name predictiq-backend
aws ecr create-repository --repository-name predictiq-frontend
```

#### 2. Build and Push Images

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker-compose build
docker tag predictiq-backend:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/predictiq-backend:latest
docker tag predictiq-frontend:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/predictiq-frontend:latest

# Push
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/predictiq-backend:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/predictiq-frontend:latest
```

#### 3. Create ECS Cluster

```bash
aws ecs create-cluster --cluster-name predictiq-cluster
```

#### 4. Set Up RDS (PostgreSQL)

```bash
aws rds create-db-instance \
  --db-instance-identifier predictiq-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username admin \
  --master-user-password YOUR_SECURE_PASSWORD \
  --allocated-storage 20
```

#### 5. Deploy with ECS Task Definitions

Create `task-definition.json` and deploy:

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster predictiq-cluster --service-name predictiq-backend --task-definition predictiq-backend --desired-count 2
```

---

### Google Cloud (Cloud Run)

#### 1. Build and Push to GCR

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/predictiq-backend ./backend
gcloud builds submit --tag gcr.io/PROJECT_ID/predictiq-frontend ./frontend
```

#### 2. Deploy Services

```bash
# Backend
gcloud run deploy predictiq-backend \
  --image gcr.io/PROJECT_ID/predictiq-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL=$DATABASE_URL,GROQ_API_KEY=$GROQ_API_KEY

# Frontend
gcloud run deploy predictiq-frontend \
  --image gcr.io/PROJECT_ID/predictiq-frontend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 3. Set Up Cloud SQL (PostgreSQL)

```bash
gcloud sql instances create predictiq-db \
  --database-version=POSTGRES_14 \
  --tier=db-f1-micro \
  --region=us-central1

gcloud sql databases create predictiq --instance=predictiq-db
```

---

### Azure (Container Instances)

#### 1. Create Resource Group

```bash
az group create --name predictiq-rg --location eastus
```

#### 2. Create Container Registry

```bash
az acr create --resource-group predictiq-rg --name predictiqacr --sku Basic
az acr login --name predictiqacr
```

#### 3. Build and Push Images

```bash
docker tag predictiq-backend predictiqacr.azurecr.io/backend:latest
docker tag predictiq-frontend predictiqacr.azurecr.io/frontend:latest

docker push predictiqacr.azurecr.io/backend:latest
docker push predictiqacr.azurecr.io/frontend:latest
```

#### 4. Deploy Container Instance

```bash
az container create \
  --resource-group predictiq-rg \
  --name predictiq-backend \
  --image predictiqacr.azurecr.io/backend:latest \
  --dns-name-label predictiq-api \
  --ports 8000 \
  --environment-variables DATABASE_URL=$DATABASE_URL GROQ_API_KEY=$GROQ_API_KEY
```

---

## 🔐 SSL/TLS Configuration

### Option 1: Let's Encrypt (Recommended)

#### Using Certbot

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal (already set up by certbot)
sudo certbot renew --dry-run
```

#### Using Docker + Certbot

Add to `docker-compose.yml`:

```yaml
certbot:
  image: certbot/certbot
  volumes:
    - ./nginx/ssl:/etc/letsencrypt
    - ./nginx/certbot:/var/www/certbot
  entrypoint: "/bin/sh -c 'trap exit TERM; while :; do certbot renew; sleep 12h & wait $${!}; done;'"
```

### Option 2: Custom SSL Certificate

```bash
# Copy your certificates
sudo cp your-cert.pem /etc/nginx/ssl/cert.pem
sudo cp your-key.pem /etc/nginx/ssl/key.pem

# Set permissions
sudo chmod 600 /etc/nginx/ssl/*.pem

# Update nginx.conf with certificate paths
sudo nano /etc/nginx/nginx.conf

# Reload Nginx
sudo nginx -s reload
```

---

## 🗄️ Database Migration

### SQLite to PostgreSQL

#### 1. Export SQLite Data

```bash
sqlite3 backend/database.db .dump > database_dump.sql
```

#### 2. Create PostgreSQL Database

```bash
psql -U postgres -c "CREATE DATABASE predictiq;"
```

#### 3. Migrate Schema

```bash
# Install pgloader
sudo apt install pgloader

# Run migration
pgloader database.db postgresql://user:password@localhost/predictiq
```

#### 4. Update Database URL

```bash
# In .env file
DATABASE_URL=postgresql://predictiq_user:password@postgres:5432/predictiq
```

#### 5. Run Migrations

```bash
cd backend
alembic upgrade head
```

---

## 📊 Monitoring Setup

### Prometheus Configuration

1. Ensure Prometheus scrapes your backend:

```yaml
# monitoring/prometheus.yml
scrape_configs:
  - job_name: 'predictiq-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

2. Add alerts (`monitoring/alerts/rules.yml`):

```yaml
groups:
  - name: predictiq_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
```

### Grafana Dashboards

1. Access Grafana: `http://your-domain:3001`
2. Import dashboard from `monitoring/grafana/dashboards/predictiq.json`
3. Configure alerts to Slack/Email

---

## 💾 Backup & Recovery

### Automated Backups

#### Database Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)

docker-compose exec -T postgres pg_dump -U predictiq_user predictiq | gzip > "$BACKUP_DIR/backup_$DATE.sql.gz"

# Retain only last 7 days
find $BACKUP_DIR -type f -mtime +7 -delete
```

#### Schedule with Cron

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

### Recovery Procedure

```bash
# Stop services
docker-compose down

# Restore database
gunzip -c /backups/postgres/backup_YYYYMMDD_HHMMSS.sql.gz | docker-compose exec -T postgres psql -U predictiq_user predictiq

# Restart services
docker-compose up -d
```

---

## 🔧 Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs backend

# Check resource usage
docker stats

# Restart specific service
docker-compose restart backend
```

### Database Connection Issues

```bash
# Test database connection
docker-compose exec backend python -c "from db.database import engine; print(engine.connect())"

# Check PostgreSQL logs
docker-compose logs postgres
```

### Model Loading Errors

```bash
# Verify models exist
docker-compose exec backend ls -lh models/

# Re-run training pipeline
docker-compose exec backend python pipeline/model_training.py
```

### High Memory Usage

```bash
# Limit container memory in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
```

### WebSocket Connection Failures

```bash
# Check Nginx WebSocket configuration
docker-compose logs nginx

# Test WebSocket endpoint
wscat -c ws://localhost:8000/ws/realtime-predict
```

---

## 📞 Support

For deployment assistance:
- **Email**: support@predictiq.com
- **Slack**: [#deployment](https://predictiq.slack.com/deployment)
- **Issues**: [GitHub Issues](https://github.com/yourusername/predictiq/issues)

---

**Last Updated**: 2024-01-15
