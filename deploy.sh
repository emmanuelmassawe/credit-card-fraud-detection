#!/bin/bash

echo "🚀 Deploying Fraud Detection to Kubernetes..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t fraud-detection-app:latest .

# Load image to Kind/Minikube (if using local cluster)
# kind load docker-image fraud-detection-app:latest
# minikube image load fraud-detection-app:latest

# Apply Kubernetes manifests
echo "☸️  Applying Kubernetes manifests..."
kubectl apply -k k8s/

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/fraud-detection-app -n fraud-detection

# Get service info
echo "✅ Deployment complete!"
echo ""
echo "📊 Service Info:"
kubectl get svc -n fraud-detection
echo ""
echo "🔍 Pod Status:"
kubectl get pods -n fraud-detection
echo ""
echo "🌐 Access the application:"
kubectl get svc fraud-detection-loadbalancer -n fraud-detection