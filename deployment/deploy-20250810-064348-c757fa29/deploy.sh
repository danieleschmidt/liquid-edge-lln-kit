#!/bin/bash
# Liquid Edge LLN Production Deployment Script
# Deployment ID: deploy-20250810-064348-c757fa29
# Version: 1.0.0

set -euo pipefail

echo "🚀 Starting Liquid Edge LLN deployment..."
echo "Deployment ID: deploy-20250810-064348-c757fa29"
echo "Version: 1.0.0"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
kubectl cluster-info > /dev/null || { echo "❌ Kubernetes cluster not accessible"; exit 1; }
docker --version > /dev/null || { echo "❌ Docker not available"; exit 1; }

# Build and push container image
echo "📦 Building container image..."
docker build -t ghcr.io/liquid-edge/liquid-edge-lln:latest .
docker push ghcr.io/liquid-edge/liquid-edge-lln:latest

# Deploy to Kubernetes
echo "⚙️ Deploying to Kubernetes..."
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-ingress.yaml

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/liquid-edge-lln -n production --timeout=300s

# Health check
echo "🏥 Running post-deployment health checks..."
kubectl get pods -n production -l app=liquid-edge-lln

# Setup monitoring
echo "📊 Setting up monitoring..."
kubectl apply -f prometheus.yml
echo "Grafana dashboard available at: grafana-dashboard.json"

echo "✅ Deployment completed successfully!"
echo "🌐 Application URL: https://api.liquid-edge.ai"
echo "📊 Monitoring: https://grafana.liquid-edge.ai"
