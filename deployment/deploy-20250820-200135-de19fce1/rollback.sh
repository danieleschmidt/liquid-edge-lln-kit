#!/bin/bash
# Liquid Edge LLN Rollback Script
# Deployment ID: deploy-20250820-200135-de19fce1

set -euo pipefail

echo "🔄 Starting rollback procedure..."

# Get previous deployment
PREVIOUS_VERSION=$(kubectl get deployment liquid-edge-lln -n production -o jsonpath='{.metadata.annotations.previous-version}')

if [ -z "$PREVIOUS_VERSION" ]; then
    echo "❌ No previous version found for rollback"
    exit 1
fi

echo "📈 Rolling back to version: $PREVIOUS_VERSION"

# Rollback deployment
kubectl rollout undo deployment/liquid-edge-lln -n production

# Wait for rollback to complete
kubectl rollout status deployment/liquid-edge-lln -n production --timeout=300s

# Verify rollback
kubectl get pods -n production -l app=liquid-edge-lln

echo "✅ Rollback completed successfully!"
echo "Current version: $PREVIOUS_VERSION"
