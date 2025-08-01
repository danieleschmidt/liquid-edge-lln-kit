# Model Deployment Runbook

## Problem Description
Deploy a new liquid neural network model to production hardware or update an existing deployment.

## Pre-Deployment Checklist

### Model Validation
- [ ] Model passes all unit tests
- [ ] Energy consumption within budget (< target mW)
- [ ] Inference latency meets requirements (< target ms)
- [ ] Accuracy meets minimum threshold (> target %)
- [ ] Model tested on target hardware platform

### Deployment Readiness
- [ ] Target devices accessible and healthy
- [ ] Backup of current model created
- [ ] Rollback procedure tested
- [ ] Monitoring dashboards configured
- [ ] Stakeholders notified of deployment window

## Deployment Procedures

### 1. Simulation Deployment
```bash
# Test deployment in simulation first
liquid-lln deploy --target simulation --model new_model.liquid
liquid-lln verify --target simulation --test-suite full
```

### 2. Staging Deployment
```bash
# Deploy to staging hardware
liquid-lln deploy --target stm32h743 --model new_model.liquid --env staging
liquid-lln verify --target stm32h743 --test-suite integration
```

### 3. Production Deployment

#### Blue-Green Deployment (Recommended)
```bash
# Deploy to green environment
liquid-lln deploy --target production-green --model new_model.liquid

# Verify green environment
liquid-lln verify --target production-green --test-suite production

# Switch traffic to green
liquid-lln switch --from production-blue --to production-green

# Monitor for 15 minutes
sleep 900

# Decommission blue environment
liquid-lln decommission --target production-blue
```

#### Rolling Deployment
```bash
# Deploy to subset of devices
liquid-lln deploy --target production --model new_model.liquid --subset 25%

# Monitor performance
liquid-lln monitor --target production --duration 300

# Continue rolling deployment
liquid-lln deploy --target production --model new_model.liquid --subset 100%
```

### 4. Verification Steps

#### Functional Verification
```bash
# Test basic functionality
curl -X POST http://device-ip/inference -d '{"input": [1,2,3,4]}'

# Test end-to-end pipeline
liquid-lln test --target production --scenario end-to-end
```

#### Performance Verification
```bash
# Check energy consumption
liquid-lln profile --target production --metric energy --duration 60

# Check inference latency
liquid-lln profile --target production --metric latency --samples 1000

# Check accuracy
liquid-lln evaluate --target production --dataset validation
```

#### Health Verification
```bash
# Check system health
liquid-lln health --target production

# Check resource usage
liquid-lln monitor --target production --metric system --duration 300
```

## Monitoring During Deployment

### Key Metrics to Watch
- **Inference Latency**: Should remain < baseline + 10%
- **Energy Consumption**: Should remain within budget
- **Error Rate**: Should remain < 1%
- **Memory Usage**: Should not exceed 80% of available
- **Device Temperature**: Should remain within safe range

### Automated Alerts
Ensure these alerts are active during deployment:
- High error rate (> 5%)
- High latency (> threshold + 20%)
- High energy consumption (> budget + 15%)
- Device unresponsive
- Memory exhaustion

## Rollback Procedures

### Automatic Rollback Triggers
- Error rate > 10% for 2 minutes
- Average latency > threshold + 50%
- Energy consumption > budget + 25%
- Device crash or unresponsive

### Manual Rollback
```bash
# Emergency rollback
liquid-lln rollback --target production --to-previous

# Rollback to specific version
liquid-lln rollback --target production --to-version v1.2.3

# Verify rollback success
liquid-lln verify --target production --test-suite smoke
```

## Post-Deployment Tasks

### Monitoring Setup
- [ ] Update monitoring dashboards with new model metrics
- [ ] Configure alerts for new model parameters
- [ ] Document baseline performance metrics
- [ ] Set up automated reporting

### Documentation Updates
- [ ] Update deployment logs
- [ ] Document any issues encountered
- [ ] Update model registry
- [ ] Share deployment summary with team

### Performance Baseline
```bash
# Establish new performance baselines
liquid-lln benchmark --target production --duration 3600 --output baseline.json

# Update monitoring thresholds
liquid-lln monitor --update-thresholds --baseline baseline.json
```

## Troubleshooting Common Issues

### Deployment Failures

#### Model Upload Fails
```bash
# Check device connectivity
ping device-ip

# Check available storage
liquid-lln status --target device-ip --metric storage

# Retry with smaller batch size
liquid-lln deploy --target device-ip --model new_model.liquid --batch-size 1
```

#### Model Crashes on Device
```bash
# Check device logs
liquid-lln logs --target device-ip --follow

# Check memory usage
liquid-lln status --target device-ip --metric memory

# Deploy debug version
liquid-lln deploy --target device-ip --model new_model.liquid --debug
```

#### Performance Degradation
```bash
# Profile current performance
liquid-lln profile --target device-ip --detailed

# Compare with baseline
liquid-lln compare --target device-ip --baseline previous_baseline.json

# Analyze bottlenecks
liquid-lln analyze --target device-ip --metric performance
```

## Escalation Procedures

### Escalation Triggers
- Rollback fails
- Multiple devices become unresponsive
- Critical functionality broken
- Data corruption detected

### Escalation Contacts
1. **On-call Engineer**: Immediate response for critical issues
2. **Hardware Team**: For device-specific problems
3. **ML Team**: For model accuracy issues
4. **DevOps Team**: For infrastructure problems

### Emergency Procedures
1. **STOP** all ongoing deployments
2. **ISOLATE** affected devices from production traffic
3. **NOTIFY** incident response team
4. **DOCUMENT** all actions taken
5. **ESCALATE** to on-call engineer

## Deployment Checklist

### Pre-Deployment
- [ ] Model validated and tested
- [ ] Deployment procedure reviewed
- [ ] Rollback plan prepared
- [ ] Monitoring configured
- [ ] Team notified

### During Deployment
- [ ] Monitor key metrics continuously
- [ ] Document any issues
- [ ] Verify each step completion
- [ ] Check automated alerts

### Post-Deployment
- [ ] Performance verified
- [ ] Monitoring updated
- [ ] Documentation updated
- [ ] Team notified of completion
- [ ] Lessons learned documented