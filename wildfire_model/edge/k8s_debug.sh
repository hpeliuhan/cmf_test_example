#!/bin/bash
"""
Kubernetes Deployment Diagnostic Script
"""

echo "🔧 Kubernetes Wildfire Deployment Diagnostics"
echo "=============================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

echo "📋 Checking deployment status..."
kubectl get pods -l app=wifire-al

echo ""
echo "📊 Checking persistent volumes..."
kubectl get pvc | grep wildfire

echo ""
echo "🔍 Checking pod details..."
POD_NAME=$(kubectl get pods -l app=wifire-al -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD_NAME" ]; then
    echo "❌ No wifire-al pods found"
    echo "   Run: kubectl apply -f edge.yaml"
    exit 1
fi

echo "Pod: $POD_NAME"

echo ""
echo "📱 Checking pod environment variables..."
kubectl exec $POD_NAME -- env | grep -E "(MODEL_CACHE_DIR|FINE_TUNE_SERVER_URL|UPLOAD_FOLDER)" || echo "❌ Environment variables not set"

echo ""
echo "📁 Checking mounted volumes..."
kubectl exec $POD_NAME -- ls -la /app/models || echo "❌ Cannot access /app/models"
kubectl exec $POD_NAME -- ls -la /app/videos || echo "❌ Cannot access /app/videos"  
kubectl exec $POD_NAME -- ls -la /app/results || echo "❌ Cannot access /app/results"

echo ""
echo "🤖 Checking for model files..."
kubectl exec $POD_NAME -- find /app -name "*.h5" 2>/dev/null || echo "❌ No .h5 files found"

echo ""
echo "📜 Recent pod logs..."
kubectl logs $POD_NAME --tail=20

echo ""
echo "🔗 Testing health endpoint..."
kubectl exec $POD_NAME -- curl -s http://localhost:5000/health 2>/dev/null || echo "❌ Health endpoint not responding"

echo ""
echo "🎯 Testing model status endpoint..."
kubectl exec $POD_NAME -- curl -s http://localhost:5000/get_current_model_status 2>/dev/null | head -c 200 || echo "❌ Model status endpoint not responding"

echo ""
echo "💾 Checking disk usage..."
kubectl exec $POD_NAME -- df -h | grep app

echo ""
echo "🔧 Pod resource usage..."
kubectl top pod $POD_NAME 2>/dev/null || echo "❌ Metrics not available"

echo ""
echo "=============================================="
echo "💡 Common Kubernetes Issues:"
echo "1. Environment variables not injected (missing envFrom)"
echo "2. Volume permission issues (need init container)"
echo "3. Insufficient memory for model loading"
echo "4. PVC not properly mounted"
echo "5. Pod restart losing in-memory state"
echo ""
echo "🔧 To apply fixed configuration:"
echo "   kubectl apply -f edge.yaml"
echo "   kubectl rollout restart deployment/wifire-al"
