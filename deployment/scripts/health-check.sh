#!/bin/bash
set -euo pipefail

# Comprehensive health check script
echo "🏥 Running comprehensive health checks"

NAMESPACE="liquid-edge"
SERVICE_NAME="liquid-edge-service"

# Check pod status
echo "📋 Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=liquid-edge

# Check service status
echo "📋 Checking service status..."
kubectl get service $SERVICE_NAME -n $NAMESPACE

# Get service endpoint
CLUSTER_IP=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
PORT=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.spec.ports[0].port}')

if [ ! -z "$CLUSTER_IP" ] && [ ! -z "$PORT" ]; then
    # Port forward for testing
    kubectl port-forward service/$SERVICE_NAME 8080:$PORT -n $NAMESPACE &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    # Run health checks
    echo "🔍 Testing health endpoint..."
    curl -f http://localhost:8080/health || echo "Health endpoint failed"
    
    echo "🔍 Testing ready endpoint..."
    curl -f http://localhost:8080/ready || echo "Ready endpoint failed"
    
    echo "🔍 Testing metrics endpoint..."
    curl -f http://localhost:8080/metrics || echo "Metrics endpoint failed"
    
    # Kill port forward
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    echo "✅ Health checks completed"
else
    echo "❌ Could not determine service endpoint"
fi
