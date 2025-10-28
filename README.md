# Intelligent Adaptive Cyber Defense Framework for Securing Renewable Energy Grids

## A Hybrid AI-Driven Approach for Industry 4.0

This repository contains the complete implementation of the Adaptive Cyber Defense Framework as described in the research paper "Intelligent Adaptive Cyber Defense Framework for Securing Renewable Energy Grids: A Hybrid AI-Driven Approach for Industry 4.0".

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Real-time Monitoring](#real-time-monitoring)
- [Results](#results)
- [File Structure](#file-structure)
- [Citation](#citation)
- [License](#license)

---

## 🔍 Overview

The framework addresses cybersecurity challenges in modern renewable energy grids by integrating **four complementary AI paradigms**:

1. **XGBoost** - Supervised learning for known threat classification
2. **Variational Autoencoder (VAE)** - Unsupervised learning for zero-day anomaly detection
3. **Graph Convolutional Network (GCN)** - Graph-based modeling of grid topology
4. **Deep Q-Network (DQN)** - Reinforcement learning for autonomous threat mitigation

### Performance Highlights

- ✅ **94.5% Detection Accuracy**
- ✅ **2.3% False Positive Rate**
- ✅ **0.85s Response Time**
- ✅ **91.6% Grid Stability Maintenance**
- ✅ **87.4% Mitigation Efficiency**
- ✅ **29%+ improvement** over conventional systems

---

## 🚀 Key Features

### Multi-Paradigm AI Integration
- Combines supervised, unsupervised, and reinforcement learning
- Handles both known threats and zero-day exploits
- Adapts to evolving attack patterns

### Real-time Detection
- Sub-second response time
- Continuous monitoring capability
- Automated threat mitigation

### Grid-Aware Security
- Graph-based topology modeling
- Vulnerability assessment
- Maintains operational stability

### Comprehensive Threat Coverage
- **25 attack types** including:
  - DoS/DDoS attacks
  - False data injection
  - SCADA manipulation
  - Ransomware
  - Port scanning
  - And 20 more threat vectors

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              INPUT: Network Traffic & Grid Data             │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼─────────┐           ┌────────▼─────────┐
│  XGBoost        │           │   VAE            │
│  Classifier     │           │   Anomaly        │
│  (Supervised)   │           │   Detector       │
└───────┬─────────┘           └────────┬─────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                ┌───────▼──────────┐
                │  GCN Graph       │
                │  Topology        │
                │  Analyzer        │
                └───────┬──────────┘
                        │
                ┌───────▼──────────┐
                │  DQN Threat      │
                │  Mitigator       │
                │  (RL Agent)      │
                └───────┬──────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼─────────┐           ┌────────▼─────────┐
│  Threat         │           │  Mitigation      │
│  Detection      │           │  Actions         │
│  Results        │           │  & Response      │
└─────────────────┘           └──────────────────┘
```

---

## 💾 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM recommended

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/adaptive-cyber-defense.git
cd adaptive-cyber-defense
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```python
python -c "import tensorflow, torch, xgboost; print('Installation successful!')"
```

---

## 📊 Dataset

The framework uses a comprehensive dataset of **40,000 cyber-attack instances** spanning **25 threat vectors**.

### Dataset Generation

```python
from dataset_generator import RenewableEnergyGridDatasetGenerator

# Generate dataset
generator = RenewableEnergyGridDatasetGenerator(seed=42)
dataset = generator.generate_dataset(
    total_samples=40000,
    save_path='renewable_energy_grid_dataset.csv'
)
```

### Dataset Features

**Network Features (10):**
- Packet rate, byte rate, protocol anomaly score
- Connection frequency, port scan indicator
- Payload entropy, source/destination diversity
- Temporal pattern deviation, flow duration

**Control System Features (5):**
- Command frequency, setpoint deviation
- Response time, authentication failures
- Protocol violations

**Physical System Features (5):**
- Power output, voltage level, frequency
- Temperature, vibration level

**Attack Types (25):**
```
Normal, DoS Attack, DDoS Attack, Port Scan, Vulnerability Scan,
Brute Force, SQL Injection, Command Injection, Man-in-the-Middle,
ARP Spoofing, DNS Spoofing, Replay Attack, False Data Injection,
SCADA Manipulation, PLC Tampering, Firmware Modification,
Ransomware, Malware Infection, Trojan, Worm Propagation,
Zero Day Exploit, Privilege Escalation, Lateral Movement,
Data Exfiltration, Resource Exhaustion
```

---

## 🔧 Usage

### Quick Start

```python
from adaptive_cyber_defense_framework import AdaptiveCyberDefenseFramework
import pandas as pd

# Initialize framework
framework = AdaptiveCyberDefenseFramework()

# Load trained models (if available)
framework.load_models('trained_models/')

# Load test data
test_data = pd.read_csv('dataset_test.csv')
X_test = test_data[feature_columns].values

# Detect threats
results = framework.detect_threats(X_test)

print(f"Threats detected: {results['supervised_predictions']}")
print(f"Threat scores: {results['combined_threat_scores']}")
```

---

## 🎓 Training

### Complete Training Pipeline

```bash
python train_framework.py
```

This script:
1. ✓ Generates/loads the dataset (40,000 samples)
2. ✓ Splits into train/validation/test sets
3. ✓ Trains XGBoost classifier
4. ✓ Trains VAE anomaly detector
5. ✓ Trains GCN topology analyzer
6. ✓ Trains DQN mitigation agent
7. ✓ Evaluates performance
8. ✓ Generates visualizations
9. ✓ Saves trained models

### Training Individual Components

```python
from adaptive_cyber_defense_framework import AdaptiveCyberDefenseFramework

framework = AdaptiveCyberDefenseFramework()

# Train supervised component
framework.train_supervised_component(X_train, y_train, X_val, y_val)

# Train unsupervised component (on normal traffic only)
X_normal = X_train[y_train == 'Normal']
framework.train_unsupervised_component(X_normal, epochs=100)

# Train RL component
from train_framework import RenewableEnergyGridEnvironment
env = RenewableEnergyGridEnvironment()
framework.train_rl_component(env, episodes=1000)
```

### Training Time

Approximate training times on standard hardware:
- XGBoost: ~2-3 minutes
- VAE: ~5-10 minutes
- GCN: ~3-5 minutes
- DQN: ~15-20 minutes
- **Total: ~25-40 minutes**

---

## 📈 Evaluation

### Comprehensive Evaluation

```python
from evaluation_visualization import generate_evaluation_report

# Generate complete evaluation report
metrics = generate_evaluation_report(
    framework=framework,
    X_test=X_test,
    y_test=y_test,
    output_dir='evaluation_results/'
)
```

### Generated Outputs

1. **ROC Curves** (`roc_curves.png`)
2. **Precision-Recall Curves** (`pr_curves.png`)
3. **Confusion Matrix** (`confusion_matrix.png`)
4. **Feature Importance** (`feature_importance.png`)
5. **Performance Comparison** (`performance_comparison.png`)
6. **Classification Report** (`classification_report.csv`)
7. **Evaluation Metrics** (`evaluation_metrics.json`)

### Key Metrics

```python
# Access individual metrics
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
```

---

## 🔴 Real-time Monitoring

### Live Monitoring Simulation

```python
from realtime_monitoring import simulate_real_time_monitoring
import pandas as pd

# Load test data
test_data = pd.read_csv('dataset_test.csv')

# Run monitoring simulation
monitor = simulate_real_time_monitoring(
    framework=framework,
    test_data=test_data,
    duration_seconds=60,
    samples_per_batch=10,
    delay=1.0
)

# Get statistics
monitor.print_statistics()
```

### Custom Monitoring

```python
from realtime_monitoring import RealTimeMonitor

# Initialize monitor
monitor = RealTimeMonitor(framework, alert_threshold=0.7)

# Monitor traffic batch
results = monitor.monitor_traffic(traffic_batch)

# Access alerts
for alert in results['alerts']:
    print(f"Threat: {alert['threat_type']}, Severity: {alert['severity']}")

# Access mitigation recommendations
for mitigation in results['mitigation_recommendations']:
    print(f"Action: {mitigation['recommended_action']}")
```

### Mitigation Actions

The DQN agent recommends from 10 possible actions:

0. **Monitor** - Continue monitoring without intervention
1. **Block Traffic** - Block suspicious network traffic
2. **Isolate Node** - Isolate affected grid node
3. **Security Patch** - Apply security patch to vulnerable component
4. **Increase Monitoring** - Enhance monitoring level
5. **Reset Component** - Reset compromised component
6. **Firewall Rules** - Enable additional firewall rules
7. **Backup Activation** - Activate backup system
8. **Subsystem Shutdown** - Shut down affected subsystem
9. **Grid Reconfiguration** - Emergency grid reconfiguration

---

## 📊 Results

### Comparison with State-of-the-Art Methods

| Method | Detection Accuracy | FPR | Response Time |
|--------|-------------------|-----|---------------|
| **Proposed Framework** | **94.5%** | **2.3%** | **0.85s** |
| Deep Learning-IDS | 91.2% | 3.8% | 1.2s |
| ML-Ensemble | 89.5% | 4.5% | 1.5s |
| Graph-CNN | 88.7% | 5.2% | 1.8s |
| LSTM-Autoencoder | 87.3% | 5.8% | 2.1s |
| Random Forest | 85.6% | 6.5% | 2.5s |
| SVM-RBF | 82.1% | 8.2% | 3.2s |
| Isolation Forest | 80.5% | 9.1% | 3.8s |
| CNN-LSTM | 79.8% | 10.3% | 4.5s |
| Gradient Boosting | 78.2% | 11.2% | 5.1s |
| Rule-based System | 65.3% | 18.5% | 8.2s |

### Performance Improvements

- **29.2%** improvement in grid stability maintenance
- **29.1%** improvement in mitigation efficiency
- **~50%** reduction in false positive rate vs. traditional methods
- **~90%** faster response time vs. rule-based systems

---

## 📁 File Structure

```
adaptive-cyber-defense/
│
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
│
├── adaptive_cyber_defense_framework.py    # Main framework implementation
├── dataset_generator.py                   # Dataset generation
├── train_framework.py                     # Training pipeline
├── evaluation_visualization.py            # Evaluation and plots
├── realtime_monitoring.py                 # Real-time monitoring
│
├── trained_models/                        # Saved model weights
│   ├── xgboost_model.pkl
│   ├── vae_weights.h5
│   ├── gcn_weights.pth
│   └── dqn_weights.h5
│
├── datasets/                              # Generated datasets
│   ├── renewable_energy_grid_dataset.csv
│   ├── dataset_train.csv
│   ├── dataset_validation.csv
│   └── dataset_test.csv
│
├── results/                               # Evaluation results
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── feature_importance.png
│   ├── performance_comparison.png
│   ├── classification_report.csv
│   └── evaluation_metrics.json
│
└── monitoring_logs/                       # Real-time monitoring logs
    ├── alerts.csv
    ├── detection_log.csv
    ├── mitigation_actions.csv
    └── monitoring_statistics.json
```

---

## 🎯 Key Components

### 1. XGBoost Threat Classifier

```python
# Supervised learning for known threats
classifier = XGBoostThreatClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1
)
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)
```

### 2. Variational Autoencoder

```python
# Unsupervised anomaly detection
vae = VariationalAutoencoder(
    input_dim=15,
    latent_dim=8,
    hidden_dims=[64, 32]
)
vae.train(X_normal, epochs=100)
anomalies = vae.detect_anomalies(X_test)
```

### 3. Graph Convolutional Network

```python
# Grid topology analysis
gcn = GraphConvolutionalNetwork(
    input_dim=20,
    hidden_dims=[64, 32],
    output_dim=16
)
vulnerability_scores = gcn.compute_vulnerability_score(
    node_features, edge_index
)
```

### 4. Deep Q-Network Agent

```python
# Reinforcement learning for mitigation
dqn = DQNAgent(
    state_dim=30,
    action_dim=10,
    hidden_dims=[128, 64]
)
action = dqn.select_action(current_state)
```

---

## 🔬 Research Paper Details

**Title:** Intelligent Adaptive Cyber Defense Framework for Securing Renewable Energy Grids: A Hybrid AI-Driven Approach for Industry 4.0

**Dataset:** 40,000 cyber-attack instances across 25 threat vectors

**Key Contributions:**
1. Novel hybrid AI framework combining SL, UL, and RL
2. Real-time graph-based grid modeling
3. Multi-objective optimization framework
4. Comprehensive evaluation methodology
5. Superior performance vs. 10 state-of-the-art methods

---

## 🛠️ Advanced Usage

### Custom Configuration

```python
# Define custom configuration
custom_config = {
    'xgboost_n_estimators': 300,
    'xgboost_max_depth': 12,
    'vae_latent_dim': 16,
    'dqn_hidden_dims': [256, 128, 64]
}

# Initialize framework with custom config
framework = AdaptiveCyberDefenseFramework(config=custom_config)
```

### Batch Processing

```python
# Process large datasets in batches
batch_size = 1000
results_list = []

for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    batch_results = framework.detect_threats(batch)
    results_list.append(batch_results)
```

### Model Fine-tuning

```python
# Load pre-trained models and fine-tune
framework.load_models('trained_models/')

# Fine-tune on new data
framework.train_supervised_component(
    X_new, y_new, epochs=50
)

# Save updated models
framework.save_models('trained_models_finetuned/')
```

---

## 📝 Logging and Debugging

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run your code with detailed logs
framework.train_supervised_component(X_train, y_train)
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```python
# Solution: Reduce batch size
framework.train_unsupervised_component(X_normal, batch_size=16)
```

**Issue: Import errors**
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Issue: Model training too slow**
```python
# Solution: Use CPU/GPU appropriately
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration:
- **Email:** fmhharbi@taibahu.edu.sa
- **Issues:** [GitHub Issues](https://github.com/FatemahAlharbi/ai-gridshield/issues)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Research supported by Taibah University, Saudi Arabia with grant number (1008-13-447)
- Dataset inspired by real-world renewable energy grid architectures
- Thanks to the open-source ML/DL community

---

## 📊 Performance Benchmarks

Tested on:
- **CPU:** Intel Core i7-10700K
- **GPU:** NVIDIA RTX 3080
- **RAM:** 32GB DDR4
- **OS:** Ubuntu 20.04 LTS

Training Time: ~30 minutes
Inference Time: ~0.85 seconds per batch (1000 samples)

---

## 🔮 Future Work

- [ ] Integration with real SCADA systems
- [ ] Extended threat vector coverage
- [ ] Federated learning implementation
- [ ] Edge deployment optimization
- [ ] Mobile/IoT device support

---

## ⭐ Star History

If you find this project useful, please consider giving it a star!

---

**Last Updated:** October 2025
**Version:** 1.0.0
**Status:** Production Ready ✅

---

Made with ❤️ for securing renewable energy infrastructure
