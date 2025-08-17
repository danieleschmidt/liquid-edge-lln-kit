#!/usr/bin/env python3
import sys
import time
import requests

def health_check():
    try:
        # Check main service
        response = requests.get('http://localhost:8080/health', timeout=5)
        if response.status_code != 200:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
        
        health_data = response.json()
        
        # Check quantum coherence
        if health_data.get('quantum_coherence', 0) < 0.5:
            print(f"Quantum coherence too low: {health_data.get('quantum_coherence')}")
            return False
        
        # Check system health
        if health_data.get('system_health') == 'failed':
            print("System health check failed")
            return False
        
        print("Health check passed")
        return True
        
    except Exception as e:
        print(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    if health_check():
        sys.exit(0)
    else:
        sys.exit(1)
