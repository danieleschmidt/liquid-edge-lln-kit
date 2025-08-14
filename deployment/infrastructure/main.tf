# Terraform configuration for Liquid Edge LLN production deployment
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
