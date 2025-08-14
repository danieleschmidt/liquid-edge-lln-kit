#!/bin/bash
set -euo pipefail

# Liquid Edge LLN Rollback Script
echo "⏪ Starting rollback process"

ENVIRONMENT=${1:-production}
NAMESPACE="liquid-edge"

# Get previous revision
PREVIOUS_REVISION=$(kubectl rollout history deployment/liquid-edge-deployment -n $NAMESPACE | tail -n 2 | head -n 1 | awk '{print $1}')

if [ -z "$PREVIOUS_REVISION" ]; then
    echo "❌ No previous revision found"
    exit 1
fi

echo "Rolling back to revision: $PREVIOUS_REVISION"

# Perform rollback
kubectl rollout undo deployment/liquid-edge-deployment --to-revision=$PREVIOUS_REVISION -n $NAMESPACE

# Wait for rollback to complete
kubectl rollout status deployment/liquid-edge-deployment -n $NAMESPACE --timeout=300s

echo "✅ Rollback completed successfully"
