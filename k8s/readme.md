# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (Minikube, Kind, GKE, EKS, AKS)
- kubectl installed
- Docker installed

## Quick Start

### 1. Setup Minikube (Local Development)

\`\`\`bash
# Start Minikube
minikube start --cpus=4 --memory=8192

# Enable addons
minikube addons enable metrics-server
minikube addons enable ingress
\`\`\`

### 2. Build Docker Image

\`\`\`bash
docker build -t fraud-detection-app:latest .
\`\`\`

### 3. Load Image to Minikube

\`\`\`bash
minikube image load fraud-detection-app:latest
\`\`\`

### 4. Deploy to Kubernetes

\`\`\`bash
# Apply all manifests
kubectl apply -k k8s/

# Or use the script
./deploy.sh
\`\`\`

### 5. Check Deployment

\`\`\`bash
# Check pods
kubectl get pods -n fraud-detection

# Check services
kubectl get svc -n fraud-detection

# Check logs
kubectl logs -f -n fraud-detection -l app=fraud-detection
\`\`\`

### 6. Access Application

\`\`\`bash
# Get service URL (Minikube)
minikube service fraud-detection-loadbalancer -n fraud-detection --url

# Port forward (alternative)
kubectl port-forward -n fraud-detection svc/fraud-detection-service 8000:80
\`\`\`

## Scaling

\`\`\`bash
# Manual scaling
kubectl scale deployment fraud-detection-app -n fraud-detection --replicas=5

# Auto-scaling is configured via HPA
kubectl get hpa -n fraud-detection
\`\`\`

## Update Deployment

\`\`\`bash
# Update image
kubectl set image deployment/fraud-detection-app -n fraud-detection \
  fraud-detection=fraud-detection-app:v2

# Rollout status
kubectl rollout status deployment/fraud-detection-app -n fraud-detection

# Rollback
kubectl rollout undo deployment/fraud-detection-app -n fraud-detection
\`\`\`

## Monitoring

\`\`\`bash
# View metrics
kubectl top pods -n fraud-detection
kubectl top nodes

# View HPA status
kubectl get hpa -n fraud-detection -w
\`\`\`

## Cleanup

\`\`\`bash
# Delete all resources
kubectl delete -k k8s/

# Or delete namespace
kubectl delete namespace fraud-detection
\`\`\`
\`\`\`

---

## Step 13 - Test Locally with Minikube

```bash
# Install Minikube (if not installed)
# Windows: choco install minikube
# Mac: brew install minikube
# Linux: curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

# Start Minikube
minikube start --cpus=4 --memory=8192

# Enable metrics-server for HPA
minikube addons enable metrics-server

# Build image
docker build -t fraud-detection-app:latest .

# Load image to Minikube
minikube image load fraud-detection-app:latest

# Deploy
kubectl apply -k k8s/

# Check status
kubectl get all -n fraud-detection

# Access application
minikube service fraud-detection-loadbalancer -n fraud-detection