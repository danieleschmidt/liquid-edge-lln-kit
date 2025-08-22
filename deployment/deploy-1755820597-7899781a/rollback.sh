#!/bin/bash
set -euo pipefail

echo "🔄 Rolling back Liquid Edge LLN deployment..."

# Rollback deployment
kubectl rollout undo deployment/liquid-edge-lln

# Wait for rollback
kubectl rollout status deployment/liquid-edge-lln --timeout=300s

# Verify rollback
kubectl get pods -l app=liquid-edge-lln

echo "✅ Rollback completed successfully!"
