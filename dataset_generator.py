"""
Dataset Generator for Renewable Energy Grid Cyber Attacks
Generates comprehensive synthetic dataset with 40,000 attack instances
spanning 25 threat vectors as described in the research paper

Features include:
- Network traffic patterns
- Control system behaviors
- Physical system parameters
- Temporal patterns
- 25 different attack types
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime, timedelta


class RenewableEnergyGridDatasetGenerator:
    """
    Generate comprehensive cyber attack dataset for renewable energy grids
    """
    
    def __init__(self, seed=42):
        """Initialize dataset generator"""
        np.random.seed(seed)
        
        # Attack types (25 threat vectors)
        self.attack_types = [
            'Normal',  # 0 - Benign traffic
            'DoS_Attack',  # 1
            'DDoS_Attack',  # 2
            'Port_Scan',  # 3
            'Vulnerability_Scan',  # 4
            'Brute_Force',  # 5
            'SQL_Injection',  # 6
            'Command_Injection',  # 7
            'Man_in_the_Middle',  # 8
            'ARP_Spoofing',  # 9
            'DNS_Spoofing',  # 10
            'Replay_Attack',  # 11
            'False_Data_Injection',  # 12
            'SCADA_Manipulation',  # 13
            'PLC_Tampering',  # 14
            'Firmware_Modification',  # 15
            'Ransomware',  # 16
            'Malware_Infection',  # 17
            'Trojan',  # 18
            'Worm_Propagation',  # 19
            'Zero_Day_Exploit',  # 20
            'Privilege_Escalation',  # 21
            'Lateral_Movement',  # 22
            'Data_Exfiltration',  # 23
            'Resource_Exhaustion'  # 24
        ]
        
        # Protocol types
        self.protocols = ['TCP', 'UDP', 'ICMP', 'IEC61850', 'DNP3', 'Modbus']
        
        # Component types in renewable energy grid
        self.components = [
            'Wind_Turbine', 'Solar_Panel', 'Battery_Storage', 
            'Inverter', 'Transformer', 'Controller', 
            'SCADA_System', 'HMI', 'RTU', 'IED'
        ]
        
    def generate_normal_traffic(self, n_samples):
        """Generate normal/benign traffic patterns"""
        data = []
        
        for _ in range(n_samples):
            sample = {
                # Network features
                'packet_rate': np.random.normal(500, 100),
                'byte_rate': np.random.normal(1500, 300),
                'protocol_anomaly_score': np.random.uniform(0, 0.2),
                'connection_frequency': np.random.poisson(10),
                'port_scan_indicator': np.random.uniform(0, 0.1),
                'payload_entropy': np.random.normal(3.5, 0.5),
                'source_diversity': np.random.randint(1, 5),
                'destination_diversity': np.random.randint(1, 5),
                'temporal_pattern_deviation': np.random.uniform(0, 0.15),
                'flow_duration': np.random.exponential(30),
                
                # Control system features
                'command_frequency': np.random.poisson(5),
                'setpoint_deviation': np.random.uniform(0, 0.1),
                'response_time': np.random.normal(50, 10),
                'authentication_failures': 0,
                'protocol_violations': 0,
                
                # Physical system features
                'power_output': np.random.normal(1000, 100),
                'voltage_level': np.random.normal(230, 5),
                'frequency': np.random.normal(60, 0.1),
                'temperature': np.random.normal(25, 3),
                'vibration_level': np.random.normal(2, 0.5),
                
                # Metadata
                'protocol': np.random.choice(self.protocols),
                'component_type': np.random.choice(self.components),
                'attack_type': 'Normal',
                'severity': 0
            }
            data.append(sample)
        
        return data
    
    def generate_dos_attack(self, n_samples):
        """Generate Denial of Service attack patterns"""
        data = []
        
        for _ in range(n_samples):
            sample = {
                'packet_rate': np.random.normal(10000, 2000),  # Very high
                'byte_rate': np.random.normal(50000, 10000),  # Very high
                'protocol_anomaly_score': np.random.uniform(0.7, 1.0),
                'connection_frequency': np.random.poisson(100),
                'port_scan_indicator': np.random.uniform(0, 0.3),
                'payload_entropy': np.random.normal(2.0, 0.5),  # Low entropy
                'source_diversity': 1,  # Single source
                'destination_diversity': 1,  # Single target
                'temporal_pattern_deviation': np.random.uniform(0.6, 1.0),
                'flow_duration': np.random.exponential(5),
                
                'command_frequency': np.random.poisson(50),
                'setpoint_deviation': np.random.uniform(0.3, 0.8),
                'response_time': np.random.normal(500, 100),  # Degraded
                'authentication_failures': np.random.poisson(20),
                'protocol_violations': np.random.poisson(15),
                
                'power_output': np.random.normal(500, 150),  # Degraded
                'voltage_level': np.random.normal(220, 15),
                'frequency': np.random.normal(59.5, 0.5),
                'temperature': np.random.normal(35, 5),
                'vibration_level': np.random.normal(4, 1),
                
                'protocol': np.random.choice(self.protocols),
                'component_type': np.random.choice(self.components),
                'attack_type': 'DoS_Attack',
                'severity': np.random.randint(7, 10)
            }
            data.append(sample)
        
        return data
    
    def generate_port_scan(self, n_samples):
        """Generate port scanning attack patterns"""
        data = []
        
        for _ in range(n_samples):
            sample = {
                'packet_rate': np.random.normal(1000, 200),
                'byte_rate': np.random.normal(500, 100),  # Small packets
                'protocol_anomaly_score': np.random.uniform(0.4, 0.7),
                'connection_frequency': np.random.poisson(50),
                'port_scan_indicator': np.random.uniform(0.8, 1.0),  # High
                'payload_entropy': np.random.normal(1.5, 0.3),  # Very low
                'source_diversity': 1,
                'destination_diversity': np.random.randint(10, 50),  # Many ports
                'temporal_pattern_deviation': np.random.uniform(0.5, 0.8),
                'flow_duration': np.random.exponential(2),
                
                'command_frequency': np.random.poisson(2),
                'setpoint_deviation': np.random.uniform(0, 0.2),
                'response_time': np.random.normal(60, 15),
                'authentication_failures': np.random.poisson(5),
                'protocol_violations': np.random.poisson(3),
                
                'power_output': np.random.normal(950, 120),
                'voltage_level': np.random.normal(228, 7),
                'frequency': np.random.normal(59.9, 0.2),
                'temperature': np.random.normal(26, 3),
                'vibration_level': np.random.normal(2.2, 0.6),
                
                'protocol': 'TCP',
                'component_type': np.random.choice(self.components),
                'attack_type': 'Port_Scan',
                'severity': np.random.randint(3, 6)
            }
            data.append(sample)
        
        return data
    
    def generate_false_data_injection(self, n_samples):
        """Generate false data injection attack patterns"""
        data = []
        
        for _ in range(n_samples):
            sample = {
                'packet_rate': np.random.normal(600, 150),
                'byte_rate': np.random.normal(2000, 400),
                'protocol_anomaly_score': np.random.uniform(0.5, 0.9),
                'connection_frequency': np.random.poisson(15),
                'port_scan_indicator': np.random.uniform(0, 0.2),
                'payload_entropy': np.random.normal(4.5, 0.8),  # High entropy
                'source_diversity': np.random.randint(1, 3),
                'destination_diversity': np.random.randint(1, 3),
                'temporal_pattern_deviation': np.random.uniform(0.6, 0.9),
                'flow_duration': np.random.exponential(25),
                
                'command_frequency': np.random.poisson(20),
                'setpoint_deviation': np.random.uniform(0.5, 1.0),  # Large deviation
                'response_time': np.random.normal(100, 30),
                'authentication_failures': np.random.poisson(2),
                'protocol_violations': np.random.poisson(8),
                
                'power_output': np.random.normal(1200, 200),  # Abnormal readings
                'voltage_level': np.random.normal(240, 10),
                'frequency': np.random.normal(60.5, 0.3),
                'temperature': np.random.normal(30, 5),
                'vibration_level': np.random.normal(3, 0.8),
                
                'protocol': np.random.choice(['IEC61850', 'DNP3', 'Modbus']),
                'component_type': np.random.choice(self.components),
                'attack_type': 'False_Data_Injection',
                'severity': np.random.randint(8, 10)
            }
            data.append(sample)
        
        return data
    
    def generate_scada_manipulation(self, n_samples):
        """Generate SCADA system manipulation attack patterns"""
        data = []
        
        for _ in range(n_samples):
            sample = {
                'packet_rate': np.random.normal(800, 150),
                'byte_rate': np.random.normal(3000, 500),
                'protocol_anomaly_score': np.random.uniform(0.6, 1.0),
                'connection_frequency': np.random.poisson(12),
                'port_scan_indicator': np.random.uniform(0, 0.1),
                'payload_entropy': np.random.normal(5.0, 0.7),
                'source_diversity': np.random.randint(1, 2),
                'destination_diversity': np.random.randint(3, 8),
                'temporal_pattern_deviation': np.random.uniform(0.7, 1.0),
                'flow_duration': np.random.exponential(40),
                
                'command_frequency': np.random.poisson(30),  # High command rate
                'setpoint_deviation': np.random.uniform(0.7, 1.0),
                'response_time': np.random.normal(200, 50),
                'authentication_failures': np.random.poisson(10),
                'protocol_violations': np.random.poisson(12),
                
                'power_output': np.random.normal(700, 200),  # Unstable
                'voltage_level': np.random.normal(225, 12),
                'frequency': np.random.normal(59.3, 0.4),
                'temperature': np.random.normal(32, 6),
                'vibration_level': np.random.normal(3.5, 1),
                
                'protocol': np.random.choice(['IEC61850', 'DNP3', 'Modbus']),
                'component_type': np.random.choice(['SCADA_System', 'HMI', 'RTU']),
                'attack_type': 'SCADA_Manipulation',
                'severity': 10
            }
            data.append(sample)
        
        return data
    
    def generate_ransomware(self, n_samples):
        """Generate ransomware attack patterns"""
        data = []
        
        for _ in range(n_samples):
            sample = {
                'packet_rate': np.random.normal(700, 150),
                'byte_rate': np.random.normal(5000, 1000),  # High data transfer
                'protocol_anomaly_score': np.random.uniform(0.6, 0.9),
                'connection_frequency': np.random.poisson(8),
                'port_scan_indicator': np.random.uniform(0.1, 0.3),
                'payload_entropy': np.random.normal(6.5, 0.5),  # Very high (encrypted)
                'source_diversity': np.random.randint(1, 3),
                'destination_diversity': np.random.randint(5, 15),
                'temporal_pattern_deviation': np.random.uniform(0.5, 0.8),
                'flow_duration': np.random.exponential(60),
                
                'command_frequency': np.random.poisson(25),
                'setpoint_deviation': np.random.uniform(0.4, 0.8),
                'response_time': np.random.normal(300, 80),
                'authentication_failures': np.random.poisson(15),
                'protocol_violations': np.random.poisson(10),
                
                'power_output': np.random.normal(0, 50),  # System shutdown
                'voltage_level': np.random.normal(210, 20),
                'frequency': np.random.normal(58, 1),
                'temperature': np.random.normal(40, 8),
                'vibration_level': np.random.normal(5, 1.5),
                
                'protocol': np.random.choice(self.protocols),
                'component_type': np.random.choice(self.components),
                'attack_type': 'Ransomware',
                'severity': 10
            }
            data.append(sample)
        
        return data
    
    def generate_dataset(self, total_samples=40000, save_path=None):
        """
        Generate complete dataset with specified distribution
        
        Args:
            total_samples: Total number of samples to generate
            save_path: Path to save the dataset (optional)
        
        Returns:
            Pandas DataFrame with generated dataset
        """
        print(f"Generating dataset with {total_samples} samples...")
        
        # Distribution of samples across attack types
        # 30% Normal, 70% Attacks distributed across 24 attack types
        n_normal = int(total_samples * 0.30)
        n_attacks = total_samples - n_normal
        
        # Generate normal traffic
        data = self.generate_normal_traffic(n_normal)
        print(f"Generated {n_normal} normal traffic samples")
        
        # Generate DoS attacks (10% of attacks)
        n_dos = int(n_attacks * 0.10)
        data.extend(self.generate_dos_attack(n_dos))
        print(f"Generated {n_dos} DoS attack samples")
        
        # Generate port scans (8% of attacks)
        n_port_scan = int(n_attacks * 0.08)
        data.extend(self.generate_port_scan(n_port_scan))
        print(f"Generated {n_port_scan} Port Scan samples")
        
        # Generate false data injection (12% of attacks)
        n_fdi = int(n_attacks * 0.12)
        data.extend(self.generate_false_data_injection(n_fdi))
        print(f"Generated {n_fdi} False Data Injection samples")
        
        # Generate SCADA manipulation (10% of attacks)
        n_scada = int(n_attacks * 0.10)
        data.extend(self.generate_scada_manipulation(n_scada))
        print(f"Generated {n_scada} SCADA Manipulation samples")
        
        # Generate ransomware (8% of attacks)
        n_ransomware = int(n_attacks * 0.08)
        data.extend(self.generate_ransomware(n_ransomware))
        print(f"Generated {n_ransomware} Ransomware samples")
        
        # Generate remaining attack types
        remaining_attacks = n_attacks - (n_dos + n_port_scan + n_fdi + n_scada + n_ransomware)
        remaining_types = [t for t in self.attack_types[2:] if t not in 
                          ['DoS_Attack', 'Port_Scan', 'False_Data_Injection', 
                           'SCADA_Manipulation', 'Ransomware', 'Normal']]
        
        samples_per_type = remaining_attacks // len(remaining_types)
        
        for attack_type in remaining_types:
            attack_data = self._generate_generic_attack(attack_type, samples_per_type)
            data.extend(attack_data)
            print(f"Generated {samples_per_type} {attack_type} samples")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add timestamp
        base_time = datetime(2024, 1, 1)
        df['timestamp'] = [base_time + timedelta(seconds=i*10) for i in range(len(df))]
        
        print(f"\nDataset generation complete!")
        print(f"Total samples: {len(df)}")
        print(f"Attack type distribution:\n{df['attack_type'].value_counts()}")
        
        # Save if path provided
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"\nDataset saved to: {save_path}")
            
            # Save metadata
            metadata = {
                'total_samples': len(df),
                'attack_types': df['attack_type'].value_counts().to_dict(),
                'features': list(df.columns),
                'generation_date': datetime.now().isoformat(),
                'description': 'Renewable Energy Grid Cyber Attack Dataset'
            }
            
            metadata_path = save_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved to: {metadata_path}")
        
        return df
    
    def _generate_generic_attack(self, attack_type, n_samples):
        """Generate generic attack pattern for remaining attack types"""
        data = []
        
        severity_map = {
            'DDoS_Attack': 9, 'Vulnerability_Scan': 4, 'Brute_Force': 6,
            'SQL_Injection': 8, 'Command_Injection': 9, 'Man_in_the_Middle': 9,
            'ARP_Spoofing': 7, 'DNS_Spoofing': 7, 'Replay_Attack': 7,
            'PLC_Tampering': 10, 'Firmware_Modification': 10,
            'Malware_Infection': 9, 'Trojan': 9, 'Worm_Propagation': 8,
            'Zero_Day_Exploit': 10, 'Privilege_Escalation': 9,
            'Lateral_Movement': 8, 'Data_Exfiltration': 9, 'Resource_Exhaustion': 7
        }
        
        for _ in range(n_samples):
            # Base pattern with random variations
            sample = {
                'packet_rate': np.random.normal(1000, 300),
                'byte_rate': np.random.normal(3000, 800),
                'protocol_anomaly_score': np.random.uniform(0.5, 0.95),
                'connection_frequency': np.random.poisson(20),
                'port_scan_indicator': np.random.uniform(0.1, 0.6),
                'payload_entropy': np.random.normal(4.0, 1.0),
                'source_diversity': np.random.randint(1, 10),
                'destination_diversity': np.random.randint(1, 10),
                'temporal_pattern_deviation': np.random.uniform(0.4, 0.9),
                'flow_duration': np.random.exponential(35),
                
                'command_frequency': np.random.poisson(15),
                'setpoint_deviation': np.random.uniform(0.3, 0.7),
                'response_time': np.random.normal(150, 50),
                'authentication_failures': np.random.poisson(8),
                'protocol_violations': np.random.poisson(6),
                
                'power_output': np.random.normal(800, 180),
                'voltage_level': np.random.normal(225, 10),
                'frequency': np.random.normal(59.7, 0.3),
                'temperature': np.random.normal(28, 4),
                'vibration_level': np.random.normal(2.8, 0.8),
                
                'protocol': np.random.choice(self.protocols),
                'component_type': np.random.choice(self.components),
                'attack_type': attack_type,
                'severity': severity_map.get(attack_type, 7)
            }
            data.append(sample)
        
        return data


def generate_training_test_split(df, test_size=0.2, val_size=0.1):
    """
    Split dataset into training, validation, and test sets
    
    Args:
        df: Complete dataset DataFrame
        test_size: Fraction for test set
        val_size: Fraction for validation set (from training data)
    
    Returns:
        Dictionary with train, val, and test DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    # First split: train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df['attack_type']
    )
    
    # Second split: train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), 
        random_state=42, stratify=train_val_df['attack_type']
    )
    
    print("\nDataset split:")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return {
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }


if __name__ == "__main__":
    # Generate dataset
    generator = RenewableEnergyGridDatasetGenerator(seed=42)
    
    # Generate 40,000 samples as specified in the paper
    dataset = generator.generate_dataset(
        total_samples=40000,
        save_path='/home/claude/renewable_energy_grid_dataset.csv'
    )
    
    print("\n" + "="*60)
    print("Dataset Statistics:")
    print("="*60)
    print(dataset.describe())
    
    print("\n" + "="*60)
    print("Sample Data:")
    print("="*60)
    print(dataset.head(10))
    
    # Generate train/val/test splits
    splits = generate_training_test_split(dataset)
    
    # Save splits
    splits['train'].to_csv('/home/claude/dataset_train.csv', index=False)
    splits['validation'].to_csv('/home/claude/dataset_validation.csv', index=False)
    splits['test'].to_csv('/home/claude/dataset_test.csv', index=False)
    
    print("\nDataset splits saved successfully!")
