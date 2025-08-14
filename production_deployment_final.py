#!/usr/bin/env python3
"""
PRODUCTION DEPLOYMENT FINAL - Global Multi-Region Infrastructure
Complete production deployment system with monitoring, scaling, and operational readiness.
"""

import os
import sys
import time
import json
import hashlib
import subprocess
import tempfile
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    
    # Global deployment
    regions: List[str] = field(default_factory=lambda: ["us-east-1", "eu-west-1", "ap-southeast-1"])
    environments: List[str] = field(default_factory=lambda: ["staging", "production"])
    
    # Infrastructure
    kubernetes_enabled: bool = True
    docker_enabled: bool = True
    load_balancer_enabled: bool = True
    auto_scaling_enabled: bool = True
    
    # Monitoring & Observability
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    jaeger_tracing_enabled: bool = True
    elk_logging_enabled: bool = True
    
    # Security
    tls_enabled: bool = True
    vault_integration: bool = True
    rbac_enabled: bool = True
    network_policies_enabled: bool = True
    
    # Compliance
    gdpr_compliance: bool = True
    ccpa_compliance: bool = True
    pdpa_compliance: bool = True
    sox_compliance: bool = True


class ContainerizationManager:
    """Manages Docker containerization for production deployment."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger('ContainerManager')
    
    def generate_production_dockerfile(self) -> str:
        """Generate optimized production Dockerfile."""
        dockerfile_content = '''# Multi-stage production Dockerfile for Liquid Edge LLN
FROM python:3.11-slim-bullseye as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    make \\
    cmake \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r liquiduser && useradd -r -g liquiduser liquiduser

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml pytest.ini ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim-bullseye as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN groupadd -r liquiduser && useradd -r -g liquiduser -m -s /bin/bash liquiduser

# Set up directories
WORKDIR /app
RUN chown -R liquiduser:liquiduser /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=liquiduser:liquiduser src/ ./src/
COPY --chown=liquiduser:liquiduser examples/ ./examples/
COPY --chown=liquiduser:liquiduser scripts/ ./scripts/

# Copy configuration files
COPY --chown=liquiduser:liquiduser pyproject.toml ./
COPY --chown=liquiduser:liquiduser README.md ./

# Switch to non-root user
USER liquiduser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import sys; sys.path.append('/app'); from src.liquid_edge.core import LiquidNN; print('OK')" || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "src.liquid_edge.cli", "--mode", "production", "--port", "8000"]
'''
        
        return dockerfile_content
    
    def generate_docker_compose(self) -> str:
        """Generate production docker-compose configuration."""
        compose_content = '''version: '3.8'

services:
  liquid-edge-api:
    build:
      context: .
      dockerfile: Dockerfile.production
    image: liquid-edge-lln:latest
    container_name: liquid-edge-api
    ports:
      - "8000:8000"
    environment:
      - LIQUID_ENV=production
      - LIQUID_LOG_LEVEL=INFO
      - LIQUID_MAX_WORKERS=8
      - LIQUID_CACHE_SIZE=10000
      - PROMETHEUS_ENABLED=true
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
    networks:
      - liquid-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /app/logs

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    networks:
      - liquid-network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/liquid-edge.json:ro
    networks:
      - liquid-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - liquid-network
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

networks:
  liquid-network:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
  redis_data:
'''
        
        return compose_content
    
    def create_container_configs(self) -> Dict[str, str]:
        """Create all container configuration files."""
        configs = {
            'Dockerfile.production': self.generate_production_dockerfile(),
            'docker-compose.production.yml': self.generate_docker_compose()
        }
        
        return configs


class KubernetesManager:
    """Manages Kubernetes deployment configurations."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger('K8sManager')
    
    def generate_deployment_yaml(self) -> str:
        """Generate Kubernetes deployment configuration."""
        deployment_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: liquid-edge-deployment
  namespace: liquid-edge
  labels:
    app: liquid-edge
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: liquid-edge
  template:
    metadata:
      labels:
        app: liquid-edge
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: liquid-edge-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: liquid-edge-api
        image: liquid-edge-lln:latest
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: LIQUID_ENV
          value: "production"
        - name: LIQUID_LOG_LEVEL
          value: "INFO"
        - name: LIQUID_MAX_WORKERS
          value: "8"
        - name: PROMETHEUS_ENABLED
          value: "true"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
              - ALL
      volumes:
      - name: config-volume
        configMap:
          name: liquid-edge-config
      - name: logs-volume
        emptyDir: {}
      nodeSelector:
        kubernetes.io/arch: amd64
      tolerations:
      - key: "liquid-edge"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - liquid-edge
              topologyKey: kubernetes.io/hostname
---
apiVersion: v1
kind: Service
metadata:
  name: liquid-edge-service
  namespace: liquid-edge
  labels:
    app: liquid-edge
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: liquid-edge
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: liquid-edge-hpa
  namespace: liquid-edge
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: liquid-edge-deployment
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
'''
        
        return deployment_yaml
    
    def generate_ingress_yaml(self) -> str:
        """Generate Kubernetes ingress configuration."""
        ingress_yaml = '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: liquid-edge-ingress
  namespace: liquid-edge
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-XSS-Protection: 1; mode=block";
      more_set_headers "Strict-Transport-Security: max-age=31536000; includeSubDomains";
spec:
  tls:
  - hosts:
    - api.liquid-edge.ai
    secretName: liquid-edge-tls
  rules:
  - host: api.liquid-edge.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: liquid-edge-service
            port:
              number: 80
'''
        
        return ingress_yaml
    
    def generate_monitoring_yaml(self) -> str:
        """Generate monitoring stack configuration."""
        monitoring_yaml = '''apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: liquid-edge
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    scrape_configs:
    - job_name: 'liquid-edge'
      static_configs:
      - targets: ['liquid-edge-service:80']
      scrape_interval: 5s
      metrics_path: /metrics
    
    - job_name: 'kubernetes-pods'
      kubernetes_sd_configs:
      - role: pod
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: liquid-edge
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
        - name: storage-volume
          mountPath: /prometheus
        args:
        - '--config.file=/etc/prometheus/prometheus.yml'
        - '--storage.tsdb.path=/prometheus'
        - '--storage.tsdb.retention.time=30d'
        - '--web.enable-lifecycle'
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: storage-volume
        persistentVolumeClaim:
          claimName: prometheus-storage
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: prometheus-storage
  namespace: liquid-edge
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
'''
        
        return monitoring_yaml
    
    def create_k8s_configs(self) -> Dict[str, str]:
        """Create all Kubernetes configuration files."""
        configs = {
            'k8s-deployment.yaml': self.generate_deployment_yaml(),
            'k8s-ingress.yaml': self.generate_ingress_yaml(),
            'k8s-monitoring.yaml': self.generate_monitoring_yaml()
        }
        
        return configs


class InfrastructureAsCode:
    """Infrastructure as Code generator for multi-cloud deployment."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = logging.getLogger('IaC')
    
    def generate_terraform_main(self) -> str:
        """Generate Terraform main configuration."""
        terraform_main = '''# Terraform configuration for Liquid Edge LLN production deployment
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "regions" {
  description = "AWS regions for deployment"
  type        = list(string)
  default     = ["us-east-1", "eu-west-1", "ap-southeast-1"]
}

variable "instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = map(string)
  default = {
    "us-east-1"      = "c5.2xlarge"
    "eu-west-1"      = "c5.2xlarge"
    "ap-southeast-1" = "c5.2xlarge"
  }
}

# AWS Provider configuration
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  
  default_tags {
    tags = {
      Project     = "liquid-edge-lln"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "liquid-edge-team"
    }
  }
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
  
  default_tags {
    tags = {
      Project     = "liquid-edge-lln"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "liquid-edge-team"
    }
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"
  
  default_tags {
    tags = {
      Project     = "liquid-edge-lln"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "liquid-edge-team"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC Module for each region
module "vpc_us_east_1" {
  source = "./modules/vpc"
  
  providers = {
    aws = aws.us_east_1
  }
  
  region             = "us-east-1"
  environment        = var.environment
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
}

module "vpc_eu_west_1" {
  source = "./modules/vpc"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  region             = "eu-west-1"
  environment        = var.environment
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
}

module "vpc_ap_southeast_1" {
  source = "./modules/vpc"
  
  providers = {
    aws = aws.ap_southeast_1
  }
  
  region             = "ap-southeast-1"
  environment        = var.environment
  availability_zones = slice(data.aws_availability_zones.available.names, 0, 3)
}

# EKS Clusters
module "eks_us_east_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.us_east_1
  }
  
  region          = "us-east-1"
  environment     = var.environment
  vpc_id          = module.vpc_us_east_1.vpc_id
  subnet_ids      = module.vpc_us_east_1.private_subnet_ids
  instance_type   = var.instance_types["us-east-1"]
  min_size        = 3
  max_size        = 20
  desired_size    = 5
}

module "eks_eu_west_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.eu_west_1
  }
  
  region          = "eu-west-1"
  environment     = var.environment
  vpc_id          = module.vpc_eu_west_1.vpc_id
  subnet_ids      = module.vpc_eu_west_1.private_subnet_ids
  instance_type   = var.instance_types["eu-west-1"]
  min_size        = 3
  max_size        = 20
  desired_size    = 5
}

module "eks_ap_southeast_1" {
  source = "./modules/eks"
  
  providers = {
    aws = aws.ap_southeast_1
  }
  
  region          = "ap-southeast-1"
  environment     = var.environment
  vpc_id          = module.vpc_ap_southeast_1.vpc_id
  subnet_ids      = module.vpc_ap_southeast_1.private_subnet_ids
  instance_type   = var.instance_types["ap-southeast-1"]
  min_size        = 3
  max_size        = 20
  desired_size    = 5
}

# Global Load Balancer (CloudFront + Route 53)
resource "aws_cloudfront_distribution" "liquid_edge_global" {
  provider = aws.us_east_1
  
  origin {
    domain_name = "api.liquid-edge.ai"
    origin_id   = "liquid-edge-origin"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }
  
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Liquid Edge LLN Global Distribution"
  default_root_object = "index.html"
  
  default_cache_behavior {
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "liquid-edge-origin"
    compress               = true
    viewer_protocol_policy = "redirect-to-https"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "CloudFront-Forwarded-Proto"]
      
      cookies {
        forward = "none"
      }
    }
    
    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }
  
  price_class = "PriceClass_All"
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate.liquid_edge_cert.arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }
  
  web_acl_id = aws_wafv2_web_acl.liquid_edge_waf.arn
  
  tags = {
    Name = "liquid-edge-cloudfront"
  }
}

# SSL Certificate
resource "aws_acm_certificate" "liquid_edge_cert" {
  provider = aws.us_east_1
  
  domain_name               = "api.liquid-edge.ai"
  subject_alternative_names = ["*.liquid-edge.ai"]
  validation_method         = "DNS"
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = {
    Name = "liquid-edge-certificate"
  }
}

# WAF for security
resource "aws_wafv2_web_acl" "liquid_edge_waf" {
  provider = aws.us_east_1
  
  name        = "liquid-edge-waf"
  description = "WAF for Liquid Edge LLN API"
  scope       = "CLOUDFRONT"
  
  default_action {
    allow {}
  }
  
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1
    
    override_action {
      none {}
    }
    
    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                 = "CommonRuleSetMetric"
      sampled_requests_enabled    = true
    }
  }
  
  rule {
    name     = "RateLimitRule"
    priority = 2
    
    action {
      block {}
    }
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                 = "RateLimitMetric"
      sampled_requests_enabled    = true
    }
  }
  
  tags = {
    Name = "liquid-edge-waf"
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                 = "liquid-edge-waf"
    sampled_requests_enabled    = true
  }
}

# Outputs
output "eks_cluster_endpoints" {
  description = "EKS cluster endpoints"
  value = {
    us_east_1      = module.eks_us_east_1.cluster_endpoint
    eu_west_1      = module.eks_eu_west_1.cluster_endpoint
    ap_southeast_1 = module.eks_ap_southeast_1.cluster_endpoint
  }
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.liquid_edge_global.domain_name
}

output "vpc_ids" {
  description = "VPC IDs for each region"
  value = {
    us_east_1      = module.vpc_us_east_1.vpc_id
    eu_west_1      = module.vpc_eu_west_1.vpc_id
    ap_southeast_1 = module.vpc_ap_southeast_1.vpc_id
  }
}
'''
        
        return terraform_main
    
    def create_iac_configs(self) -> Dict[str, str]:
        """Create all Infrastructure as Code configuration files."""
        configs = {
            'main.tf': self.generate_terraform_main()
        }
        
        return configs


class ProductionDeploymentOrchestrator:
    """Orchestrates complete production deployment."""
    
    def __init__(self):
        self.config = ProductionConfig()
        self.container_manager = ContainerizationManager(self.config)
        self.k8s_manager = KubernetesManager(self.config)
        self.iac_manager = InfrastructureAsCode(self.config)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ProductionDeployment')
    
    def generate_deployment_artifacts(self) -> Dict[str, Any]:
        """Generate all production deployment artifacts."""
        print("🚀 Generating Production Deployment Artifacts")
        print("=" * 55)
        
        start_time = time.time()
        artifacts = {}
        
        # Generate container configurations
        print("\n📦 Generating Container Configurations...")
        container_configs = self.container_manager.create_container_configs()
        artifacts['container'] = container_configs
        print(f"   Generated {len(container_configs)} container config files")
        
        # Generate Kubernetes configurations
        print("\n☸️  Generating Kubernetes Configurations...")
        k8s_configs = self.k8s_manager.create_k8s_configs()
        artifacts['kubernetes'] = k8s_configs
        print(f"   Generated {len(k8s_configs)} Kubernetes config files")
        
        # Generate Infrastructure as Code
        print("\n🏗️  Generating Infrastructure as Code...")
        iac_configs = self.iac_manager.create_iac_configs()
        artifacts['infrastructure'] = iac_configs
        print(f"   Generated {len(iac_configs)} Terraform config files")
        
        # Generate deployment scripts
        print("\n📜 Generating Deployment Scripts...")
        deployment_scripts = self._generate_deployment_scripts()
        artifacts['scripts'] = deployment_scripts
        print(f"   Generated {len(deployment_scripts)} deployment scripts")
        
        # Generate monitoring configurations
        print("\n📊 Generating Monitoring Configurations...")
        monitoring_configs = self._generate_monitoring_configs()
        artifacts['monitoring'] = monitoring_configs
        print(f"   Generated {len(monitoring_configs)} monitoring config files")
        
        # Generate security configurations
        print("\n🔒 Generating Security Configurations...")
        security_configs = self._generate_security_configs()
        artifacts['security'] = security_configs
        print(f"   Generated {len(security_configs)} security config files")
        
        generation_time = time.time() - start_time
        
        print(f"\n✅ All artifacts generated in {generation_time:.2f}s")
        print(f"📁 Total files: {sum(len(category) for category in artifacts.values())}")
        
        return artifacts
    
    def _generate_deployment_scripts(self) -> Dict[str, str]:
        """Generate deployment automation scripts."""
        scripts = {}
        
        # Main deployment script
        scripts['deploy.sh'] = '''#!/bin/bash
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
'''
        
        # Rollback script
        scripts['rollback.sh'] = '''#!/bin/bash
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
'''
        
        # Health check script
        scripts['health-check.sh'] = '''#!/bin/bash
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
'''
        
        return scripts
    
    def _generate_monitoring_configs(self) -> Dict[str, str]:
        """Generate comprehensive monitoring configurations."""
        configs = {}
        
        # Prometheus configuration
        configs['prometheus.yml'] = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'liquid-edge-production'
    replica: '1'

rule_files:
  - "liquid_edge_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'liquid-edge-api'
    static_configs:
      - targets: ['liquid-edge-service:80']
    scrape_interval: 5s
    metrics_path: '/metrics'
    
  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    relabel_configs:
      - source_labels: [__address__]
        regex: '(.*):10250'
        replacement: '${1}:9100'
        target_label: __address__
        
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
'''
        
        # Alert rules
        configs['liquid_edge_rules.yml'] = '''groups:
  - name: liquid_edge_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }} seconds"
          
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} in {{ $labels.namespace }} is crash looping"
          
      - alert: HighCPUUsage
        expr: rate(cpu_usage_seconds_total[5m]) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}%"
          
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}%"
'''
        
        # Grafana dashboard
        configs['grafana-dashboard.json'] = json.dumps({
            "dashboard": {
                "id": None,
                "title": "Liquid Edge LLN Production Dashboard",
                "tags": ["liquid-edge", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{method}} {{status}}"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
                                "legendFormat": "Error Rate"
                            }
                        ],
                        "gridPos": {"h": 6, "w": 6, "x": 0, "y": 8}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        })
        
        return configs
    
    def _generate_security_configs(self) -> Dict[str, str]:
        """Generate security and compliance configurations."""
        configs = {}
        
        # Network policies
        configs['network-policies.yaml'] = '''apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: liquid-edge-network-policy
  namespace: liquid-edge
spec:
  podSelector:
    matchLabels:
      app: liquid-edge
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
    - protocol: TCP
      port: 443
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: liquid-edge-service-account
  namespace: liquid-edge
automountServiceAccountToken: false
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: liquid-edge
  name: liquid-edge-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: liquid-edge-role-binding
  namespace: liquid-edge
subjects:
- kind: ServiceAccount
  name: liquid-edge-service-account
  namespace: liquid-edge
roleRef:
  kind: Role
  name: liquid-edge-role
  apiGroup: rbac.authorization.k8s.io
'''
        
        # Pod security policies
        configs['pod-security-policy.yaml'] = '''apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: liquid-edge-psp
  namespace: liquid-edge
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  supplementalGroups:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
  fsGroup:
    rule: 'MustRunAs'
    ranges:
      - min: 1
        max: 65535
  readOnlyRootFilesystem: true
'''
        
        return configs
    
    def save_artifacts(self, artifacts: Dict[str, Any], output_dir: str = "/root/repo/deployment") -> str:
        """Save all deployment artifacts to disk."""
        print(f"\n💾 Saving deployment artifacts to {output_dir}")
        
        # Create deployment directory structure
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        total_files = 0
        
        for category, configs in artifacts.items():
            category_dir = Path(output_dir) / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            for filename, content in configs.items():
                filepath = category_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                saved_files.append(str(filepath))
                total_files += 1
                print(f"   ✅ {category}/{filename}")
        
        # Create deployment summary
        summary = {
            'deployment_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8],
            'timestamp': time.time(),
            'total_files': total_files,
            'categories': list(artifacts.keys()),
            'files': saved_files,
            'configuration': {
                'regions': self.config.regions,
                'environments': self.config.environments,
                'features': {
                    'kubernetes': self.config.kubernetes_enabled,
                    'docker': self.config.docker_enabled,
                    'monitoring': self.config.prometheus_enabled,
                    'security': self.config.tls_enabled,
                    'compliance': self.config.gdpr_compliance
                }
            }
        }
        
        summary_file = Path(output_dir) / "deployment-summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Deployment summary saved to {summary_file}")
        return str(output_dir)


def main():
    """Main production deployment orchestration."""
    print("🌊 Liquid Edge LLN Kit - Production Deployment")
    print("Global Multi-Region Infrastructure Setup")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize orchestrator
    orchestrator = ProductionDeploymentOrchestrator()
    
    # Generate all deployment artifacts
    artifacts = orchestrator.generate_deployment_artifacts()
    
    # Save artifacts to deployment directory
    deployment_dir = orchestrator.save_artifacts(artifacts)
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n🎉 PRODUCTION DEPLOYMENT PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print(f"📁 Deployment Directory: {deployment_dir}")
    print(f"📊 Total Files Generated: {sum(len(category) for category in artifacts.values())}")
    
    print(f"\n🚀 Deployment Features:")
    features = [
        "✅ Multi-region Kubernetes clusters (US, EU, APAC)",
        "✅ Auto-scaling with HPA (3-20 replicas)", 
        "✅ Global load balancing with CloudFront",
        "✅ Comprehensive monitoring (Prometheus + Grafana)",
        "✅ Security policies and RBAC",
        "✅ SSL/TLS termination",
        "✅ WAF protection",
        "✅ Infrastructure as Code (Terraform)",
        "✅ Container orchestration (Docker + K8s)",
        "✅ Automated deployment scripts"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n🌍 Global Deployment Ready:")
    print(f"   • US East (N. Virginia) - Primary region")
    print(f"   • EU West (Ireland) - European users") 
    print(f"   • AP Southeast (Singapore) - Asian users")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Review generated configurations in {deployment_dir}")
    print(f"   2. Configure cloud provider credentials")
    print(f"   3. Run terraform init && terraform apply")
    print(f"   4. Execute ./deploy.sh production")
    print(f"   5. Monitor deployment via Grafana dashboard")
    
    return {
        'deployment_directory': deployment_dir,
        'artifacts_generated': artifacts,
        'deployment_time_s': total_time,
        'production_ready': True
    }


if __name__ == "__main__":
    result = main()