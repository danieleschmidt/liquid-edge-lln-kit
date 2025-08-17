#!/bin/bash
set -e

# Quantum-Liquid Neural Network Deployment Script
echo "🚀 Starting Quantum-Liquid deployment..."

# Configuration
NAMESPACE="quantum-liquid"
IMAGE_TAG="${1:-latest}"
ENVIRONMENT="${2:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-flight checks
log_info "Running pre-flight checks..."

if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    log_error "docker is not installed"
    exit 1
fi

# Check cluster connectivity
if ! kubectl cluster-info &> /dev/null; then
    log_error "Cannot connect to Kubernetes cluster"
    exit 1
fi

log_info "Pre-flight checks passed"

# Create namespace if it doesn't exist
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    log_info "Creating namespace $NAMESPACE"
    kubectl create namespace $NAMESPACE
fi

# Apply configurations
log_info "Applying Kubernetes manifests..."

kubectl apply -f k8s-configmap.yaml -n $NAMESPACE
kubectl apply -f k8s-rbac.yaml -n $NAMESPACE
kubectl apply -f k8s-pod-security-policy.yaml -n $NAMESPACE
kubectl apply -f k8s-network-policy.yaml -n $NAMESPACE

# Deploy application
log_info "Deploying application..."

kubectl apply -f k8s-deployment.yaml -n $NAMESPACE
kubectl apply -f k8s-service.yaml -n $NAMESPACE
kubectl apply -f k8s-ingress.yaml -n $NAMESPACE

# Setup auto-scaling
if [ "True" = "True" ]; then
    log_info "Setting up auto-scaling..."
    kubectl apply -f k8s-hpa.yaml -n $NAMESPACE
fi

# Wait for deployment to be ready
log_info "Waiting for deployment to be ready..."
kubectl rollout status deployment/quantum-liquid -n $NAMESPACE --timeout=300s

# Verify deployment
log_info "Verifying deployment..."

READY_REPLICAS=$(kubectl get deployment quantum-liquid -n $NAMESPACE -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment quantum-liquid -n $NAMESPACE -o jsonpath='{.spec.replicas}')

if [ "$READY_REPLICAS" = "$DESIRED_REPLICAS" ]; then
    log_info "Deployment successful: $READY_REPLICAS/$DESIRED_REPLICAS replicas ready"
else
    log_error "Deployment failed: $READY_REPLICAS/$DESIRED_REPLICAS replicas ready"
    exit 1
fi

# Health check
log_info "Running health check..."

SERVICE_IP=$(kubectl get svc quantum-liquid-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$SERVICE_IP" ]; then
    if curl -f http://$SERVICE_IP/health; then
        log_info "Health check passed"
    else
        log_warn "Health check failed, but deployment continues"
    fi
else
    log_warn "LoadBalancer IP not available yet"
fi

# Setup monitoring
log_info "Setting up monitoring..."
kubectl apply -f prometheus.yml -n monitoring || log_warn "Failed to apply Prometheus config"
kubectl apply -f alert-rules.yml -n monitoring || log_warn "Failed to apply alert rules"

# Display deployment information
log_info "Deployment completed successfully!"
echo ""
echo "Deployment Information:"
echo "======================"
echo "Namespace: $NAMESPACE"
echo "Environment: $ENVIRONMENT"
echo "Image Tag: $IMAGE_TAG"
echo "Replicas: $DESIRED_REPLICAS"
echo ""
echo "Services:"
kubectl get svc -n $NAMESPACE
echo ""
echo "Pods:"
kubectl get pods -n $NAMESPACE -l app=quantum-liquid
echo ""
echo "Ingress:"
kubectl get ingress -n $NAMESPACE

log_info "Deployment script completed"
