"""
Training Script for Adaptive Cyber Defense Framework
Trains all components and evaluates performance on renewable energy grid dataset
"""

import numpy as np
import pandas as pd
import sys
import os

# Import framework components
from adaptive_cyber_defense_framework import (
    AdaptiveCyberDefenseFramework,
    create_sample_grid_graph
)

from dataset_generator import (
    RenewableEnergyGridDatasetGenerator,
    generate_training_test_split
)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time


class RenewableEnergyGridEnvironment:
    """
    Reinforcement Learning Environment for Grid Threat Mitigation
    """
    
    def __init__(self, state_dim=30, action_dim=10):
        """Initialize environment"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Initialize state with normal operating conditions
        self.state = np.random.randn(self.state_dim) * 0.1
        self.step_count = 0
        self.max_steps = 100
        
        # Simulate threat level
        self.threat_level = np.random.uniform(0, 0.3)
        
        return self.state.copy()
    
    def step(self, action):
        """
        Execute action and return next state, reward, done, info
        
        Actions:
        0: Monitor (no intervention)
        1: Block suspicious traffic
        2: Isolate affected node
        3: Apply security patch
        4: Increase monitoring level
        5: Reset compromised component
        6: Enable firewall rules
        7: Activate backup system
        8: Shut down affected subsystem
        9: Emergency grid reconfiguration
        """
        # Update step count
        self.step_count += 1
        
        # Calculate action effectiveness
        action_effectiveness = {
            0: 0.1,  # Monitoring
            1: 0.5,  # Block traffic
            2: 0.7,  # Isolate node
            3: 0.6,  # Security patch
            4: 0.3,  # Increase monitoring
            5: 0.8,  # Reset component
            6: 0.6,  # Firewall
            7: 0.7,  # Backup
            8: 0.9,  # Shutdown
            9: 0.95  # Reconfiguration
        }
        
        # Action costs (operational impact)
        action_costs = {
            0: 0.0,
            1: 0.2,
            2: 0.5,
            3: 0.3,
            4: 0.1,
            5: 0.6,
            6: 0.2,
            7: 0.4,
            8: 0.8,
            9: 0.9
        }
        
        # Update threat level based on action
        effectiveness = action_effectiveness.get(action, 0.1)
        self.threat_level = max(0, self.threat_level - effectiveness * 0.5)
        
        # Add random threat evolution
        self.threat_level += np.random.uniform(0, 0.1)
        self.threat_level = min(1.0, self.threat_level)
        
        # Update state
        self.state += np.random.randn(self.state_dim) * 0.05
        self.state[0] = self.threat_level  # First element is threat level
        
        # Calculate reward
        # Reward = threat reduction - operational cost - grid instability
        threat_reduction_reward = (1.0 - self.threat_level) * 10
        operational_cost = action_costs.get(action, 0.5) * 5
        grid_stability = max(0, 1.0 - abs(self.state[1:].sum() / self.state_dim))
        stability_reward = grid_stability * 3
        
        reward = threat_reduction_reward - operational_cost + stability_reward
        
        # Check if episode is done
        done = (self.step_count >= self.max_steps) or (self.threat_level < 0.05)
        
        # Additional info
        info = {
            'threat_level': self.threat_level,
            'grid_stability': grid_stability,
            'action_cost': action_costs.get(action, 0.5)
        }
        
        return self.state.copy(), reward, done, info


def prepare_features_and_labels(df):
    """
    Prepare features and labels from dataset
    
    Args:
        df: DataFrame with dataset
    
    Returns:
        X: Feature matrix
        y: Labels
    """
    # Select numerical features
    feature_columns = [
        'packet_rate', 'byte_rate', 'protocol_anomaly_score',
        'connection_frequency', 'port_scan_indicator', 'payload_entropy',
        'source_diversity', 'destination_diversity', 'temporal_pattern_deviation',
        'flow_duration', 'command_frequency', 'setpoint_deviation',
        'response_time', 'authentication_failures', 'protocol_violations'
    ]
    
    X = df[feature_columns].values
    y = df['attack_type'].values
    
    return X, y


def plot_training_metrics(metrics_history, save_path='training_metrics.png'):
    """Plot training metrics over time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_to_plot = [
        ('detection_accuracy', 'Detection Accuracy'),
        ('false_positive_rate', 'False Positive Rate'),
        ('response_time', 'Response Time (seconds)'),
        ('f1_score', 'F1 Score')
    ]
    
    for idx, (metric_name, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        if metric_name in metrics_history:
            ax.plot(metrics_history[metric_name], marker='o', linewidth=2)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Cyber Threat Detection', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_performance_comparison(framework_metrics, save_path='performance_comparison.png'):
    """Plot performance comparison with baseline methods"""
    # Baseline methods from paper
    methods = [
        'Proposed Framework',
        'Deep Learning-IDS',
        'ML-Ensemble',
        'Graph-CNN',
        'LSTM-Autoencoder',
        'Random Forest',
        'SVM-RBF',
        'Isolation Forest',
        'CNN-LSTM',
        'Gradient Boosting',
        'Traditional Rule-based'
    ]
    
    # Performance metrics (from paper results)
    detection_accuracy = [
        framework_metrics['detection_accuracy'] * 100,
        91.2, 89.5, 88.7, 87.3, 85.6, 82.1, 80.5, 79.8, 78.2, 65.3
    ]
    
    false_positive_rate = [
        framework_metrics['false_positive_rate'] * 100,
        3.8, 4.5, 5.2, 5.8, 6.5, 8.2, 9.1, 10.3, 11.2, 18.5
    ]
    
    response_time = [
        framework_metrics['response_time'],
        1.2, 1.5, 1.8, 2.1, 2.5, 3.2, 3.8, 4.5, 5.1, 8.2
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Detection Accuracy
    axes[0].barh(methods, detection_accuracy, color='steelblue')
    axes[0].set_xlabel('Detection Accuracy (%)', fontweight='bold')
    axes[0].set_title('Detection Accuracy Comparison', fontweight='bold', fontsize=12)
    axes[0].grid(axis='x', alpha=0.3)
    
    # False Positive Rate (lower is better)
    axes[1].barh(methods, false_positive_rate, color='coral')
    axes[1].set_xlabel('False Positive Rate (%)', fontweight='bold')
    axes[1].set_title('False Positive Rate Comparison', fontweight='bold', fontsize=12)
    axes[1].grid(axis='x', alpha=0.3)
    axes[1].invert_xaxis()  # Lower is better
    
    # Response Time (lower is better)
    axes[2].barh(methods, response_time, color='mediumseagreen')
    axes[2].set_xlabel('Response Time (seconds)', fontweight='bold')
    axes[2].set_title('Response Time Comparison', fontweight='bold', fontsize=12)
    axes[2].grid(axis='x', alpha=0.3)
    axes[2].invert_xaxis()  # Lower is better
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance comparison plot saved to {save_path}")
    plt.close()


def train_framework():
    """Main training function"""
    print("="*80)
    print("ADAPTIVE CYBER DEFENSE FRAMEWORK TRAINING")
    print("="*80)
    
    # Step 1: Generate or load dataset
    print("\n[Step 1] Loading dataset...")
    
    if not os.path.exists('/home/claude/renewable_energy_grid_dataset.csv'):
        print("Dataset not found. Generating new dataset...")
        generator = RenewableEnergyGridDatasetGenerator(seed=42)
        dataset = generator.generate_dataset(
            total_samples=40000,
            save_path='/home/claude/renewable_energy_grid_dataset.csv'
        )
    else:
        print("Loading existing dataset...")
        dataset = pd.read_csv('/home/claude/renewable_energy_grid_dataset.csv')
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Attack types: {dataset['attack_type'].nunique()}")
    
    # Step 2: Split dataset
    print("\n[Step 2] Splitting dataset...")
    
    if not all([os.path.exists(f'/home/claude/dataset_{split}.csv') 
                for split in ['train', 'validation', 'test']]):
        splits = generate_training_test_split(dataset)
        splits['train'].to_csv('/home/claude/dataset_train.csv', index=False)
        splits['validation'].to_csv('/home/claude/dataset_validation.csv', index=False)
        splits['test'].to_csv('/home/claude/dataset_test.csv', index=False)
    else:
        splits = {
            'train': pd.read_csv('/home/claude/dataset_train.csv'),
            'validation': pd.read_csv('/home/claude/dataset_validation.csv'),
            'test': pd.read_csv('/home/claude/dataset_test.csv')
        }
    
    # Prepare features and labels
    X_train, y_train = prepare_features_and_labels(splits['train'])
    X_val, y_val = prepare_features_and_labels(splits['validation'])
    X_test, y_test = prepare_features_and_labels(splits['test'])
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Step 3: Initialize framework
    print("\n[Step 3] Initializing framework...")
    
    config = {
        'xgboost_n_estimators': 200,
        'xgboost_max_depth': 10,
        'xgboost_learning_rate': 0.1,
        'vae_input_dim': 15,
        'vae_latent_dim': 8,
        'vae_hidden_dims': [64, 32],
        'gcn_input_dim': 20,
        'gcn_hidden_dims': [64, 32],
        'gcn_output_dim': 16,
        'dqn_state_dim': 30,
        'dqn_action_dim': 10,
        'dqn_hidden_dims': [128, 64]
    }
    
    framework = AdaptiveCyberDefenseFramework(config=config)
    print("Framework initialized successfully")
    
    # Step 4: Train supervised component (XGBoost)
    print("\n[Step 4] Training supervised learning component (XGBoost)...")
    start_time = time.time()
    framework.train_supervised_component(X_train, y_train, X_val, y_val)
    training_time_supervised = time.time() - start_time
    print(f"Supervised component trained in {training_time_supervised:.2f} seconds")
    
    # Step 5: Train unsupervised component (VAE)
    print("\n[Step 5] Training unsupervised learning component (VAE)...")
    # Filter normal traffic for VAE training
    normal_mask = splits['train']['attack_type'] == 'Normal'
    X_normal = X_train[normal_mask]
    print(f"Normal traffic samples for VAE: {len(X_normal)}")
    
    start_time = time.time()
    framework.train_unsupervised_component(X_normal, epochs=100, batch_size=32)
    training_time_unsupervised = time.time() - start_time
    print(f"Unsupervised component trained in {training_time_unsupervised:.2f} seconds")
    
    # Step 6: Train reinforcement learning component (DQN)
    print("\n[Step 6] Training reinforcement learning component (DQN)...")
    env = RenewableEnergyGridEnvironment(
        state_dim=config['dqn_state_dim'],
        action_dim=config['dqn_action_dim']
    )
    
    start_time = time.time()
    framework.train_rl_component(env, episodes=1000)
    training_time_rl = time.time() - start_time
    print(f"RL component trained in {training_time_rl:.2f} seconds")
    
    # Step 7: Evaluate on test set
    print("\n[Step 7] Evaluating framework on test set...")
    
    # Create sample graph for evaluation
    graph_data = create_sample_grid_graph(num_nodes=50)
    
    metrics = framework.evaluate_performance(X_test, y_test, graph_data)
    
    # Additional evaluation metrics
    print("\n[Step 8] Computing detailed metrics...")
    detection_results = framework.detect_threats(X_test, graph_data)
    y_pred = detection_results['supervised_predictions']
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Plot confusion matrix
    unique_labels = sorted(list(set(y_test) | set(y_pred)))
    plot_confusion_matrix(y_test, y_pred, unique_labels, 
                         save_path='/home/claude/confusion_matrix.png')
    
    # Plot performance comparison
    plot_performance_comparison(metrics, 
                               save_path='/home/claude/performance_comparison.png')
    
    # Step 9: Save trained models
    print("\n[Step 9] Saving trained models...")
    framework.save_models('/home/claude/trained_models')
    
    # Step 10: Generate final report
    print("\n" + "="*80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*80)
    
    print("\n=== FINAL PERFORMANCE METRICS ===")
    print(f"Detection Accuracy: {metrics['detection_accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']*100:.2f}%")
    print(f"Recall: {metrics['recall']*100:.2f}%")
    print(f"F1 Score: {metrics['f1_score']*100:.2f}%")
    print(f"False Positive Rate: {metrics['false_positive_rate']*100:.2f}%")
    print(f"Average Response Time: {metrics['response_time']*1000:.2f} ms")
    
    print("\n=== TRAINING TIME ===")
    print(f"Supervised Component: {training_time_supervised:.2f} seconds")
    print(f"Unsupervised Component: {training_time_unsupervised:.2f} seconds")
    print(f"RL Component: {training_time_rl:.2f} seconds")
    print(f"Total Training Time: {training_time_supervised + training_time_unsupervised + training_time_rl:.2f} seconds")
    
    print("\n=== OUTPUT FILES ===")
    print("1. Dataset: /home/claude/renewable_energy_grid_dataset.csv")
    print("2. Train split: /home/claude/dataset_train.csv")
    print("3. Validation split: /home/claude/dataset_validation.csv")
    print("4. Test split: /home/claude/dataset_test.csv")
    print("5. Trained models: /home/claude/trained_models/")
    print("6. Confusion matrix: /home/claude/confusion_matrix.png")
    print("7. Performance comparison: /home/claude/performance_comparison.png")
    
    return framework, metrics


if __name__ == "__main__":
    # Run training
    framework, final_metrics = train_framework()
    
    print("\n" + "="*80)
    print("All operations completed successfully!")
    print("="*80)
