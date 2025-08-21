#!/bin/bash
set -euo pipefail

echo "🏥 Checking Liquid Edge LLN health..."

# Check deployment status
kubectl get deployment liquid-edge-lln -o wide

# Check pod health
kubectl get pods -l app=liquid-edge-lln -o wide

# Check service endpoints
kubectl get endpoints liquid-edge-lln-service

# Test application health
ENDPOINT=$(kubectl get service liquid-edge-lln-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ ! -z "$ENDPOINT" ]; then
    curl -f http://$ENDPOINT/health || echo "❌ Health check failed"
else
    echo "⚠️  No external endpoint available"
fi

echo "✅ Health check completed"
