#!/usr/bin/env python3
"""
Per-Request SLO Compliance Report Generator

This script analyzes the GenAI-Perf results and creates a detailed report showing
each request's actual completion time compared to its individual SLO requirements.
"""

import json
import csv
import sys
import os
from typing import Dict, List, Tuple, Any
from pathlib import Path

def load_data(artifacts_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load performance data and SLO requirements from artifacts."""
    
    # Find the latest results directory (by name)
    artifacts_path = Path(artifacts_dir)
    result_dirs = [d for d in artifacts_path.iterdir() if d.is_dir() and 'deepseek' in d.name]
    if not result_dirs:
        raise FileNotFoundError("No GenAI-Perf result directories found")
    
    # Use the directory with highest request rate (most recent test)
    latest_dir = max(result_dirs, key=lambda x: float(x.name.split('request_rate')[-1].split('/')[0]) if 'request_rate' in x.name else 0)
    
    print(f"Using results from: {latest_dir.name}")
    
    # Load performance data
    profile_export_path = latest_dir / "profile_export.json"
    with open(profile_export_path, 'r') as f:
        perf_data = json.load(f)
    
    # Load SLO requirements
    slo_path = artifacts_path / "distributed_slo_requirements.json"
    with open(slo_path, 'r') as f:
        slo_data = json.load(f)
    
    requests = perf_data['experiments'][0]['requests']
    slo_requirements = slo_data['slo_requirements']
    
    print(f"Loaded {len(requests)} performance records and {len(slo_requirements)} SLO requirements")
    
    return requests, slo_requirements

def calculate_request_metrics(request: Dict) -> Dict[str, float]:
    """Calculate actual performance metrics for a single request."""
    
    timestamps = request['response_timestamps']
    request_start_time = request['timestamp']
    
    if not timestamps:
        return {'ttft_ms': 0, 'itl_ms': 0, 'total_latency_ms': 0}
    
    # Time to First Token (TTFT)
    ttft_ns = timestamps[0] - request_start_time
    ttft_ms = ttft_ns / 1_000_000
    
    # Inter-Token Latency (ITL) - average between consecutive tokens
    if len(timestamps) > 1:
        token_intervals = []
        for i in range(1, len(timestamps)):
            interval_ns = timestamps[i] - timestamps[i-1]
            interval_ms = interval_ns / 1_000_000
            token_intervals.append(interval_ms)
        avg_itl_ms = sum(token_intervals) / len(token_intervals)
    else:
        avg_itl_ms = 0
    
    # Total request latency
    total_latency_ns = timestamps[-1] - request_start_time
    total_latency_ms = total_latency_ns / 1_000_000
    
    return {
        'ttft_ms': ttft_ms,
        'itl_ms': avg_itl_ms,
        'total_latency_ms': total_latency_ms,
        'output_tokens': len(timestamps)
    }

def check_slo_compliance(actual_metrics: Dict[str, float], slo_req: Dict[str, float]) -> Dict[str, bool]:
    """Check if actual performance meets SLO requirements."""
    
    compliance = {}
    
    if 'time_to_first_token' in slo_req:
        compliance['ttft_compliant'] = actual_metrics['ttft_ms'] <= slo_req['time_to_first_token']
    
    if 'inter_token_latency' in slo_req:
        compliance['itl_compliant'] = actual_metrics['itl_ms'] <= slo_req['inter_token_latency']
    
    if 'request_latency' in slo_req:
        compliance['latency_compliant'] = actual_metrics['total_latency_ms'] <= slo_req['request_latency']
    
    # Overall compliance - all individual SLOs must pass
    compliance['overall_compliant'] = all(compliance.values())
    
    return compliance

def generate_compliance_report(artifacts_dir: str) -> None:
    """Generate a detailed per-request compliance report."""
    
    print("Loading data...")
    requests, slo_requirements = load_data(artifacts_dir)
    
    if len(requests) != len(slo_requirements):
        print(f"WARNING: Mismatch in data lengths: {len(requests)} requests vs {len(slo_requirements)} SLO requirements")
        min_len = min(len(requests), len(slo_requirements))
        requests = requests[:min_len]
        slo_requirements = slo_requirements[:min_len]
    
    print("Analyzing per-request compliance...")
    
    # Analyze each request
    detailed_results = []
    compliance_stats = {
        'total_requests': 0,
        'ttft_compliant': 0,
        'itl_compliant': 0,
        'overall_compliant': 0
    }
    
    for i, (request, slo_req) in enumerate(zip(requests, slo_requirements)):
        actual_metrics = calculate_request_metrics(request)
        compliance = check_slo_compliance(actual_metrics, slo_req)
        
        result = {
            'request_id': i,
            'ttft_actual_ms': actual_metrics['ttft_ms'],
            'ttft_slo_ms': slo_req.get('time_to_first_token', 0),
            'ttft_compliant': compliance.get('ttft_compliant', True),
            'itl_actual_ms': actual_metrics['itl_ms'],
            'itl_slo_ms': slo_req.get('inter_token_latency', 0),
            'itl_compliant': compliance.get('itl_compliant', True),
            'total_latency_ms': actual_metrics['total_latency_ms'],
            'output_tokens': actual_metrics['output_tokens'],
            'overall_compliant': compliance['overall_compliant']
        }
        
        detailed_results.append(result)
        
        # Update stats
        compliance_stats['total_requests'] += 1
        if compliance.get('ttft_compliant', True):
            compliance_stats['ttft_compliant'] += 1
        if compliance.get('itl_compliant', True):
            compliance_stats['itl_compliant'] += 1
        if compliance['overall_compliant']:
            compliance_stats['overall_compliant'] += 1
    
    # Calculate compliance rates
    total = compliance_stats['total_requests']
    ttft_rate = (compliance_stats['ttft_compliant'] / total) * 100 if total > 0 else 0
    itl_rate = (compliance_stats['itl_compliant'] / total) * 100 if total > 0 else 0
    overall_rate = (compliance_stats['overall_compliant'] / total) * 100 if total > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print("PER-REQUEST SLO COMPLIANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Requests: {total}")
    print(f"TTFT Compliance: {compliance_stats['ttft_compliant']}/{total} ({ttft_rate:.1f}%)")
    print(f"ITL Compliance: {compliance_stats['itl_compliant']}/{total} ({itl_rate:.1f}%)")
    print(f"Overall Compliance: {compliance_stats['overall_compliant']}/{total} ({overall_rate:.1f}%)")
    print(f"Distributed Goodput Rate: {overall_rate:.1f}%")
    
    # Save detailed CSV report
    output_path = Path(artifacts_dir) / "per_request_slo_compliance.csv"
    print(f"\nSaving detailed report to: {output_path}")
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            'request_id', 'ttft_actual_ms', 'ttft_slo_ms', 'ttft_compliant',
            'itl_actual_ms', 'itl_slo_ms', 'itl_compliant', 
            'total_latency_ms', 'output_tokens', 'overall_compliant'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_results)
    
    # Show some examples
    print(f"\nSample SLO violations:")
    violations = [r for r in detailed_results if not r['overall_compliant']]
    for i, violation in enumerate(violations[:5]):
        print(f"Request {violation['request_id']}: "
              f"TTFT {violation['ttft_actual_ms']:.1f}ms vs {violation['ttft_slo_ms']:.1f}ms SLO, "
              f"ITL {violation['itl_actual_ms']:.1f}ms vs {violation['itl_slo_ms']:.1f}ms SLO")
    
    if len(violations) > 5:
        print(f"... and {len(violations) - 5} more violations")
    
    print(f"\nDetailed per-request compliance data saved to: {output_path}")

if __name__ == "__main__":
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/genai_perf"
    
    if not os.path.exists(artifacts_dir):
        print(f"Error: Artifacts directory '{artifacts_dir}' not found")
        sys.exit(1)
    
    try:
        generate_compliance_report(artifacts_dir)
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)