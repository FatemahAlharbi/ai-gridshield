"""
Real-time Threat Detection and Monitoring
Simulates real-time monitoring and threat mitigation in renewable energy grids
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime
import json
import os


class RealTimeMonitor:
    """
    Real-time monitoring and threat detection system
    """
    
    def __init__(self, framework, alert_threshold=0.7):
        """
        Initialize real-time monitor
        
        Args:
            framework: Trained adaptive cyber defense framework
            alert_threshold: Threshold for generating alerts
        """
        self.framework = framework
        self.alert_threshold = alert_threshold
        self.alerts = []
        self.detection_log = []
        self.mitigation_actions = []
        
    def monitor_traffic(self, traffic_data, graph_data=None, verbose=True):
        """
        Monitor incoming traffic for threats
        
        Args:
            traffic_data: Real-time traffic data samples
            graph_data: Current grid topology (optional)
            verbose: Print detection results
        
        Returns:
            Detection results and recommended actions
        """
        timestamp = datetime.now()
        
        # Detect threats
        detection_results = self.framework.detect_threats(traffic_data, graph_data)
        
        # Analyze results
        supervised_preds = detection_results['supervised_predictions']
        supervised_conf = detection_results['supervised_confidence']
        unsupervised_anomalies = detection_results['unsupervised_anomalies']
        combined_scores = detection_results['combined_threat_scores']
        
        # Generate alerts for high-threat samples
        alerts = []
        for idx, (pred, conf, anom, score) in enumerate(zip(
            supervised_preds, supervised_conf, unsupervised_anomalies, combined_scores
        )):
            if score > self.alert_threshold or anom == 1:
                alert = {
                    'timestamp': timestamp.isoformat(),
                    'sample_id': idx,
                    'threat_type': pred,
                    'confidence': float(conf),
                    'anomaly_detected': bool(anom),
                    'threat_score': float(score),
                    'severity': 'HIGH' if score > 0.9 else 'MEDIUM' if score > 0.7 else 'LOW'
                }
                alerts.append(alert)
                self.alerts.append(alert)
        
        # Log detection results
        self.detection_log.append({
            'timestamp': timestamp.isoformat(),
            'total_samples': len(traffic_data),
            'threats_detected': len(alerts),
            'average_threat_score': float(np.mean(combined_scores))
        })
        
        # Recommend mitigation actions for high-severity threats
        mitigation_recommendations = []
        for alert in alerts:
            if alert['severity'] in ['HIGH', 'MEDIUM']:
                # Create state representation for DQN
                state = self._create_state_vector(alert, traffic_data[alert['sample_id']])
                
                # Get mitigation action from DQN
                action = self.framework.mitigate_threat(state)
                
                mitigation = {
                    'alert_id': alert['sample_id'],
                    'threat_type': alert['threat_type'],
                    'recommended_action': self._action_to_string(action),
                    'action_code': int(action),
                    'timestamp': timestamp.isoformat()
                }
                mitigation_recommendations.append(mitigation)
                self.mitigation_actions.append(mitigation)
        
        # Print results if verbose
        if verbose:
            self._print_monitoring_results(alerts, mitigation_recommendations)
        
        return {
            'alerts': alerts,
            'mitigation_recommendations': mitigation_recommendations,
            'detection_summary': self.detection_log[-1]
        }
    
    def _create_state_vector(self, alert, sample_data):
        """Create state vector for DQN from alert and sample data"""
        # Combine alert information with sample features
        state = np.concatenate([
            [alert['threat_score']],
            sample_data[:10],  # First 10 features
            [float(alert['anomaly_detected'])],
            np.random.randn(19)  # Additional state features (simulated)
        ])
        
        return state[:30]  # Ensure state dimension matches DQN
    
    def _action_to_string(self, action):
        """Convert action code to human-readable string"""
        action_map = {
            0: "Monitor (No immediate action)",
            1: "Block suspicious traffic",
            2: "Isolate affected node",
            3: "Apply security patch",
            4: "Increase monitoring level",
            5: "Reset compromised component",
            6: "Enable firewall rules",
            7: "Activate backup system",
            8: "Shut down affected subsystem",
            9: "Emergency grid reconfiguration"
        }
        return action_map.get(action, f"Unknown action {action}")
    
    def _print_monitoring_results(self, alerts, mitigations):
        """Print monitoring results to console"""
        print("\n" + "="*80)
        print(f"REAL-TIME MONITORING UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        if alerts:
            print(f"\n⚠️  {len(alerts)} THREAT(S) DETECTED:")
            for alert in alerts:
                print(f"  • {alert['severity']} severity - {alert['threat_type']}")
                print(f"    Threat Score: {alert['threat_score']:.3f}, Confidence: {alert['confidence']:.3f}")
        else:
            print("\n✓ No threats detected - System operating normally")
        
        if mitigations:
            print(f"\n🛡️  RECOMMENDED MITIGATION ACTIONS:")
            for mit in mitigations:
                print(f"  • {mit['threat_type']}: {mit['recommended_action']}")
        
        print("="*80)
    
    def get_monitoring_statistics(self):
        """Get comprehensive monitoring statistics"""
        if not self.detection_log:
            return None
        
        total_samples = sum([log['total_samples'] for log in self.detection_log])
        total_threats = sum([log['threats_detected'] for log in self.detection_log])
        avg_threat_score = np.mean([log['average_threat_score'] for log in self.detection_log])
        
        # Count threats by type
        threat_types = {}
        for alert in self.alerts:
            threat_type = alert['threat_type']
            threat_types[threat_type] = threat_types.get(threat_type, 0) + 1
        
        # Count mitigation actions
        action_counts = {}
        for action in self.mitigation_actions:
            action_name = action['recommended_action']
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
        
        statistics = {
            'total_samples_monitored': total_samples,
            'total_threats_detected': total_threats,
            'detection_rate': total_threats / total_samples if total_samples > 0 else 0,
            'average_threat_score': avg_threat_score,
            'threats_by_type': threat_types,
            'mitigation_actions_taken': action_counts,
            'monitoring_duration': len(self.detection_log),
            'alerts_generated': len(self.alerts)
        }
        
        return statistics
    
    def save_monitoring_logs(self, output_dir='/home/claude/monitoring_logs'):
        """Save monitoring logs and statistics"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save alerts
        if self.alerts:
            alerts_df = pd.DataFrame(self.alerts)
            alerts_df.to_csv(os.path.join(output_dir, 'alerts.csv'), index=False)
            print(f"Alerts saved to {output_dir}/alerts.csv")
        
        # Save detection log
        if self.detection_log:
            log_df = pd.DataFrame(self.detection_log)
            log_df.to_csv(os.path.join(output_dir, 'detection_log.csv'), index=False)
            print(f"Detection log saved to {output_dir}/detection_log.csv")
        
        # Save mitigation actions
        if self.mitigation_actions:
            mitigations_df = pd.DataFrame(self.mitigation_actions)
            mitigations_df.to_csv(os.path.join(output_dir, 'mitigation_actions.csv'), index=False)
            print(f"Mitigation actions saved to {output_dir}/mitigation_actions.csv")
        
        # Save statistics
        statistics = self.get_monitoring_statistics()
        if statistics:
            with open(os.path.join(output_dir, 'monitoring_statistics.json'), 'w') as f:
                json.dump(statistics, f, indent=4)
            print(f"Statistics saved to {output_dir}/monitoring_statistics.json")
    
    def print_statistics(self):
        """Print comprehensive monitoring statistics"""
        stats = self.get_monitoring_statistics()
        
        if not stats:
            print("No monitoring data available")
            return
        
        print("\n" + "="*80)
        print("MONITORING STATISTICS SUMMARY")
        print("="*80)
        
        print(f"\nTotal Samples Monitored:    {stats['total_samples_monitored']}")
        print(f"Total Threats Detected:     {stats['total_threats_detected']}")
        print(f"Detection Rate:             {stats['detection_rate']*100:.2f}%")
        print(f"Average Threat Score:       {stats['average_threat_score']:.3f}")
        print(f"Alerts Generated:           {stats['alerts_generated']}")
        
        print("\n--- Threats by Type ---")
        for threat_type, count in sorted(stats['threats_by_type'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  {threat_type:25s}: {count:4d}")
        
        print("\n--- Mitigation Actions Taken ---")
        for action, count in sorted(stats['mitigation_actions_taken'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {action:40s}: {count:4d}")
        
        print("\n" + "="*80)


def simulate_real_time_monitoring(framework, test_data, duration_seconds=60, 
                                  samples_per_batch=10, delay=1.0):
    """
    Simulate real-time monitoring scenario
    
    Args:
        framework: Trained framework
        test_data: Test dataset
        duration_seconds: Simulation duration in seconds
        samples_per_batch: Number of samples per monitoring batch
        delay: Delay between batches (seconds)
    """
    print("\n" + "="*80)
    print("STARTING REAL-TIME MONITORING SIMULATION")
    print("="*80)
    print(f"Duration: {duration_seconds} seconds")
    print(f"Samples per batch: {samples_per_batch}")
    print(f"Monitoring interval: {delay} seconds")
    print("="*80)
    
    # Initialize monitor
    monitor = RealTimeMonitor(framework, alert_threshold=0.7)
    
    # Prepare data
    feature_columns = [
        'packet_rate', 'byte_rate', 'protocol_anomaly_score',
        'connection_frequency', 'port_scan_indicator', 'payload_entropy',
        'source_diversity', 'destination_diversity', 'temporal_pattern_deviation',
        'flow_duration', 'command_frequency', 'setpoint_deviation',
        'response_time', 'authentication_failures', 'protocol_violations'
    ]
    
    X = test_data[feature_columns].values
    
    # Start simulation
    start_time = time.time()
    batch_count = 0
    
    while (time.time() - start_time) < duration_seconds:
        # Select random batch
        batch_indices = np.random.choice(len(X), size=samples_per_batch, replace=False)
        batch_data = X[batch_indices]
        
        # Monitor batch
        results = monitor.monitor_traffic(batch_data, verbose=True)
        
        batch_count += 1
        
        # Wait before next batch
        time.sleep(delay)
    
    # Print final statistics
    print("\n" + "="*80)
    print("REAL-TIME MONITORING SIMULATION COMPLETED")
    print("="*80)
    print(f"Total batches processed: {batch_count}")
    print(f"Actual duration: {time.time() - start_time:.2f} seconds")
    
    monitor.print_statistics()
    
    # Save logs
    monitor.save_monitoring_logs()
    
    return monitor


def main():
    """Main execution function for real-time monitoring demo"""
    print("Real-Time Threat Detection and Monitoring System")
    print("This script demonstrates real-time monitoring capabilities")
    print("\nTo run simulation:")
    print("1. Ensure framework is trained")
    print("2. Load test dataset")
    print("3. Call simulate_real_time_monitoring() function")


if __name__ == "__main__":
    main()
