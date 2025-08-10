# Liquid Edge LLN - Operations Runbook

## System Overview

The Liquid Edge Liquid Neural Network system provides ultra-fast, energy-efficient inference for edge robotics applications.

**System Specifications:**
- Deployment ID: `deploy-20250810-064348-c757fa29`
- Version: `1.0.0`
- Target Availability: 99.9%
- Expected Throughput: 36,836 inferences/second
- Expected Latency: <0.1ms P99

## Daily Operations

### Morning Health Check (9:00 AM)
```bash
# Check system status
kubectl get pods -n production -l app=liquid-edge-lln

# Verify performance metrics
curl https://api.liquid-edge.ai/metrics

# Check error logs
kubectl logs -l app=liquid-edge-lln -n production --since=24h | grep ERROR
```

### Load Testing (Weekly - Fridays 2:00 PM)
```bash
# Run performance validation
./performance-test.sh

# Expected results:
# - Throughput: >30,000 inf/sec
# - P99 Latency: <1ms
# - Error Rate: <0.01%
```

## Incident Response

### Severity Levels

#### P0 - Critical (Service Down)
- **Response Time**: 15 minutes
- **Escalation**: Page on-call engineer
- **Actions**: 
  1. Check pod status
  2. Review recent deployments
  3. Initiate rollback if needed
  4. Engage incident commander

#### P1 - High Impact (Performance Degraded)
- **Response Time**: 1 hour
- **Escalation**: Slack alert
- **Actions**:
  1. Analyze performance metrics
  2. Check resource utilization
  3. Scale up if needed
  4. Investigate root cause

#### P2 - Medium Impact (Minor Issues)
- **Response Time**: 4 hours
- **Escalation**: Email notification
- **Actions**:
  1. Log issue for investigation
  2. Monitor for escalation
  3. Plan fix during next maintenance

### Common Incident Scenarios

#### High Latency (P99 >100ms)
**Symptoms**: Increased response times, user complaints
**Investigation Steps**:
1. Check CPU/Memory usage: `kubectl top pods -n production`
2. Review network latency: `kubectl exec -it <pod> -- ping api.liquid-edge.ai`
3. Analyze inference queue depth
4. Check for memory leaks

**Mitigation**:
- Scale up replicas: `kubectl scale deployment liquid-edge-lln --replicas=6`
- Clear caches if needed
- Restart pods with memory leaks

#### High Error Rate (>1%)
**Symptoms**: 500 errors, failed inferences
**Investigation Steps**:
1. Check application logs: `kubectl logs -l app=liquid-edge-lln --since=1h`
2. Verify input validation errors
3. Check database connectivity
4. Review recent model changes

**Mitigation**:
- Rollback recent deployment if correlated
- Fix input validation issues
- Scale up database connections

#### Low Throughput (<10,000/sec)
**Symptoms**: Slow inference processing
**Investigation Steps**:
1. Check pod replica count
2. Verify auto-scaling configuration
3. Analyze batch processing efficiency
4. Review CPU throttling

**Mitigation**:
- Increase replica count
- Optimize batch sizes
- Adjust CPU limits
- Review load balancing

## Maintenance Procedures

### Weekly Maintenance (Sundays 2:00 AM UTC)

#### 1. System Updates
```bash
# Update container images
kubectl set image deployment/liquid-edge-lln \
  liquid-edge-lln=ghcr.io/liquid-edge/liquid-edge-lln:latest

# Wait for rollout
kubectl rollout status deployment/liquid-edge-lln -n production
```

#### 2. Performance Optimization
```bash
# Clear caches
kubectl exec -it <pod> -- python3 -c "import gc; gc.collect()"

# Restart pods (rolling restart)
kubectl rollout restart deployment/liquid-edge-lln -n production
```

#### 3. Backup Verification
```bash
# Verify model checkpoints
ls -la /backups/models/$(date +%Y-%m-%d)/

# Test restore procedure (staging)
./test-backup-restore.sh
```

### Monthly Maintenance (First Sunday of Month)

#### 1. Security Updates
- Update base container images
- Scan for vulnerabilities
- Update TLS certificates
- Review access logs

#### 2. Performance Tuning
- Analyze month-over-month metrics
- Adjust resource limits based on usage
- Optimize caching strategies
- Review scaling thresholds

#### 3. Capacity Planning
- Forecast resource needs
- Plan for traffic growth
- Evaluate hardware upgrades
- Review cost optimization

## Monitoring Dashboards

### Primary Dashboard: Grafana
- **URL**: https://grafana.liquid-edge.ai
- **Key Panels**: Throughput, Latency, Error Rate, Resource Usage

### Alert Manager
- **URL**: https://alerts.liquid-edge.ai
- **Integration**: Slack, PagerDuty, Email

### Log Analysis: ELK Stack
- **URL**: https://logs.liquid-edge.ai
- **Retention**: 30 days
- **Search**: Kibana interface

## Performance Baselines

### Normal Operating Range
- **Throughput**: 25,000 - 40,000 inferences/second
- **P99 Latency**: 0.1 - 5.0 ms
- **Error Rate**: 0.001 - 0.01%
- **CPU Usage**: 40 - 70%
- **Memory Usage**: 300 - 800 MB
- **Energy Consumption**: 1.5 - 5.0 mW

### Alert Thresholds
- **Throughput** < 15,000 inf/sec (Warning), < 10,000 inf/sec (Critical)
- **P99 Latency** > 50ms (Warning), > 100ms (Critical)
- **Error Rate** > 0.1% (Warning), > 1.0% (Critical)
- **CPU Usage** > 80% (Warning), > 90% (Critical)
- **Memory Usage** > 900MB (Warning), > 1GB (Critical)

## Contact Information

### On-Call Rotation
- **Primary**: liquid-edge-oncall@company.com
- **Secondary**: platform-oncall@company.com
- **Manager**: liquid-edge-lead@company.com

### Escalation Path
1. On-call Engineer (15 min response)
2. Team Lead (30 min response)  
3. Engineering Manager (1 hour response)
4. VP Engineering (2 hour response)

### External Support
- **Cloud Provider**: support@cloudprovider.com
- **Kubernetes**: k8s-support@company.com
- **Monitoring**: monitoring-support@company.com

---

*Last Updated: 2025-08-10T06:43:48.356702+00:00*
*Document Version: 1.0*
