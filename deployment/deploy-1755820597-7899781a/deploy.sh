#!/bin/bash
set -euo pipefail

# Production deployment script for Liquid Edge LLN Kit
# Generated: 2025-08-21 23:56:37

echo "🚀 Starting Liquid Edge LLN Production Deployment"
echo "Deployment ID: deploy-1755820597-7899781a"

# Build Docker image
echo "📦 Building production Docker image..."
docker build -t liquid-edge-lln:latest -f Dockerfile .

# Tag for registry
REGISTRY_URL="${REGISTRY_URL:-localhost:5000}"
docker tag liquid-edge-lln:latest $REGISTRY_URL/liquid-edge-lln:latest

# Push to registry
echo "📤 Pushing to container registry..."
docker push $REGISTRY_URL/liquid-edge-lln:latest

# Deploy to Kubernetes
echo "☸️  Deploying to Kubernetes..."
kubectl apply -f k8s-deployment.yaml
kubectl apply -f k8s-service.yaml
kubectl apply -f k8s-ingress.yaml

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/liquid-edge-lln --timeout=300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -l app=liquid-edge-lln
kubectl get services liquid-edge-lln-service

echo "🎉 Deployment completed successfully!"
echo "🔗 Access URL: https://liquid-edge-lln.example.com"
