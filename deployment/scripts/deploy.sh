#!/bin/bash
set -euo pipefail

# Liquid Edge LLN Production Deployment Script
echo "🚀 Starting Liquid Edge LLN Production Deployment"

# Configuration
ENVIRONMENT=${1:-production}
REGION=${2:-us-east-1}
NAMESPACE="liquid-edge"

echo "Environment: $ENVIRONMENT"
echo "Region: $REGION"
echo "Namespace: $NAMESPACE"

# Build and push Docker image
echo "📦 Building production Docker image..."
docker build -f Dockerfile.production -t liquid-edge-lln:latest .
docker tag liquid-edge-lln:latest liquid-edge-lln:$ENVIRONMENT-$(date +%Y%m%d-%H%M%S)

# ECR login and push (if using AWS)
if command -v aws &> /dev/null; then
    echo "🔐 Logging into AWS ECR..."
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
    docker push liquid-edge-lln:latest
fi

# Apply Kubernetes configurations
echo "☸️ Applying Kubernetes configurations..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy monitoring stack first
kubectl apply -f k8s-monitoring.yaml -n $NAMESPACE

# Deploy application
envsubst < k8s-deployment.yaml | kubectl apply -f - -n $NAMESPACE
kubectl apply -f k8s-ingress.yaml -n $NAMESPACE

# Wait for deployment to be ready
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/liquid-edge-deployment -n $NAMESPACE --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Run health checks
echo "🏥 Running health checks..."
EXTERNAL_IP=$(kubectl get service liquid-edge-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ ! -z "$EXTERNAL_IP" ]; then
    curl -f http://$EXTERNAL_IP/health || echo "Health check failed"
fi

echo "🎉 Deployment completed successfully!"
