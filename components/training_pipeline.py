import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional
import os

class TrainingPipelineComponent:
    """Streamlit component for model training pipeline"""
    
    def __init__(self, hybrid_predictor):
        self.hybrid_predictor = hybrid_predictor
        self.training_history = []
    
    def render(self):
        """Render the training pipeline interface"""
        # Training configuration
        self.render_training_config()
        
        # Training controls
        self.render_training_controls()
        
        # Training monitoring
        if st.session_state.get('training_status') == 'running':
            self.render_training_monitor()
        
        # Training history
        if 'training_history' in st.session_state:
            self.render_training_history()
    
    def render_training_config(self):
        """Render training configuration section"""
        st.subheader("⚙️ Training Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Model Parameters**")
            
            learning_rate = st.number_input(
                "Learning Rate:",
                min_value=1e-6,
                max_value=1e-1,
                value=1e-4,
                format="%.2e",
                key="learning_rate"
            )
            
            batch_size = st.selectbox(
                "Batch Size:",
                [8, 16, 32, 64, 128],
                index=2,
                key="batch_size"
            )
            
            epochs = st.number_input(
                "Number of Epochs:",
                min_value=1,
                max_value=1000,
                value=50,
                key="epochs"
            )
        
        with col2:
            st.write("**Data Configuration**")
            
            train_split = st.slider(
                "Training Split:",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05,
                key="train_split"
            )
            
            validation_split = st.slider(
                "Validation Split:",
                min_value=0.05,
                max_value=0.3,
                value=0.15,
                step=0.05,
                key="validation_split"
            )
            
            data_augmentation = st.checkbox(
                "Enable Data Augmentation",
                value=True,
                key="data_augmentation"
            )
        
        with col3:
            st.write("**Training Options**")
            
            mixed_precision = st.checkbox(
                "Mixed Precision Training",
                value=True,
                help="Use FP16 for faster training",
                key="mixed_precision"
            )
            
            early_stopping = st.checkbox(
                "Early Stopping",
                value=True,
                key="early_stopping"
            )
            
            save_checkpoints = st.checkbox(
                "Save Checkpoints",
                value=True,
                key="save_checkpoints"
            )
            
            wandb_logging = st.checkbox(
                "Weights & Biases Logging",
                value=False,
                help="Enable experiment tracking",
                key="wandb_logging"
            )
    
    def render_training_controls(self):
        """Render training control buttons"""
        st.subheader("🎮 Training Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Start Training", type="primary", disabled=st.session_state.get('training_status') == 'running'):
                self.start_training()
        
        with col2:
            if st.button("Pause Training", disabled=st.session_state.get('training_status') != 'running'):
                st.session_state.training_status = 'paused'
                st.success("Training paused")
        
        with col3:
            if st.button("Resume Training", disabled=st.session_state.get('training_status') != 'paused'):
                st.session_state.training_status = 'running'
                st.success("Training resumed")
        
        with col4:
            if st.button("Stop Training", disabled=st.session_state.get('training_status') not in ['running', 'paused']):
                st.session_state.training_status = 'stopped'
                st.warning("Training stopped")
    
    def render_training_monitor(self):
        """Render real-time training monitoring"""
        st.subheader("📊 Training Monitor")
        
        # Training progress
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_epoch = st.session_state.get('current_epoch', 0)
            total_epochs = st.session_state.get('epochs', 50)
            st.metric("Current Epoch", f"{current_epoch}/{total_epochs}")
        
        with col2:
            current_loss = st.session_state.get('current_loss', 0.0)
            st.metric("Training Loss", f"{current_loss:.4f}")
        
        with col3:
            val_loss = st.session_state.get('validation_loss', 0.0)
            st.metric("Validation Loss", f"{val_loss:.4f}")
        
        with col4:
            learning_rate = st.session_state.get('current_lr', st.session_state.get('learning_rate', 1e-4))
            st.metric("Learning Rate", f"{learning_rate:.2e}")
        
        # Progress bar
        if total_epochs > 0:
            progress = current_epoch / total_epochs
            st.progress(progress)
        
        # Real-time loss plot
        if 'loss_history' in st.session_state:
            self.plot_realtime_losses()
    
    def render_training_history(self):
        """Render training history and results"""
        st.subheader("📈 Training History")
        
        history = st.session_state.training_history
        
        if not history:
            st.info("No training history available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Plot training curves
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss curves
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['epoch'],
                y=df['train_loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            
            if 'val_loss' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['epoch'],
                    y=df['val_loss'],
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='red')
                ))
            
            fig.update_layout(
                title="Training Loss Curves",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Metrics curves
            if 'accuracy' in df.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=df['epoch'],
                    y=df['accuracy'],
                    mode='lines',
                    name='Training Accuracy',
                    line=dict(color='green')
                ))
                
                if 'val_accuracy' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df['epoch'],
                        y=df['val_accuracy'],
                        mode='lines',
                        name='Validation Accuracy',
                        line=dict(color='orange')
                    ))
                
                fig.update_layout(
                    title="Accuracy Curves",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Training summary table
        st.write("**Training Summary**")
        st.dataframe(df.tail(10), use_container_width=True)
        
        # Best metrics
        if len(df) > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_train_loss = df['train_loss'].min()
                st.metric("Best Training Loss", f"{best_train_loss:.4f}")
            
            with col2:
                if 'val_loss' in df.columns:
                    best_val_loss = df['val_loss'].min()
                    st.metric("Best Validation Loss", f"{best_val_loss:.4f}")
            
            with col3:
                if 'accuracy' in df.columns:
                    best_accuracy = df['accuracy'].max()
                    st.metric("Best Accuracy", f"{best_accuracy:.4f}")
    
    def start_training(self):
        """Start the training process"""
        st.session_state.training_status = 'running'
        st.session_state.current_epoch = 0
        st.session_state.loss_history = []
        
        # Initialize training parameters
        config = {
            'learning_rate': st.session_state.learning_rate,
            'batch_size': st.session_state.batch_size,
            'epochs': st.session_state.epochs,
            'train_split': st.session_state.train_split,
            'validation_split': st.session_state.validation_split,
            'mixed_precision': st.session_state.mixed_precision,
            'early_stopping': st.session_state.early_stopping
        }
        
        # Initialize Weights & Biases if enabled
        if st.session_state.get('wandb_logging', False):
            self.init_wandb(config)
        
        # Show training start message
        st.success("Training started!")
        st.info("Training is running in simulation mode. In a real implementation, this would start the actual training loop.")
        
        # Simulate training progress
        self.simulate_training(config)
    
    def simulate_training(self, config: Dict):
        """Simulate training progress for demonstration"""
        epochs = config['epochs']
        
        # Initialize training history
        history = []
        
        # Simulate training epochs
        for epoch in range(min(5, epochs)):  # Simulate first 5 epochs
            # Simulate training metrics
            train_loss = 1.0 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.05)
            val_loss = 1.1 * np.exp(-epoch * 0.15) + np.random.normal(0, 0.07)
            accuracy = 0.5 + 0.4 * (1 - np.exp(-epoch * 0.3)) + np.random.normal(0, 0.02)
            val_accuracy = 0.45 + 0.4 * (1 - np.exp(-epoch * 0.25)) + np.random.normal(0, 0.03)
            
            # Add to history
            epoch_data = {
                'epoch': epoch + 1,
                'train_loss': max(0.01, train_loss),
                'val_loss': max(0.01, val_loss),
                'accuracy': min(0.99, max(0.01, accuracy)),
                'val_accuracy': min(0.99, max(0.01, val_accuracy)),
                'learning_rate': config['learning_rate'] * (0.95 ** epoch)
            }
            
            history.append(epoch_data)
            
            # Update session state
            st.session_state.current_epoch = epoch + 1
            st.session_state.current_loss = epoch_data['train_loss']
            st.session_state.validation_loss = epoch_data['val_loss']
            st.session_state.current_lr = epoch_data['learning_rate']
        
        # Store history
        st.session_state.training_history = history
        st.session_state.training_status = 'completed'
        
        st.success("Training simulation completed!")
    
    def plot_realtime_losses(self):
        """Plot real-time loss curves"""
        loss_history = st.session_state.get('loss_history', [])
        
        if not loss_history:
            return
        
        epochs = list(range(1, len(loss_history) + 1))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=loss_history,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Real-time Training Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def init_wandb(self, config: Dict):
        """Initialize Weights & Biases logging"""
        try:
            # In a real implementation, this would initialize wandb
            st.info("Weights & Biases logging initialized (simulation mode)")
            
            # Log configuration
            st.write("**Logged Configuration:**")
            for key, value in config.items():
                st.write(f"- {key}: {value}")
                
        except Exception as e:
            st.error(f"Failed to initialize W&B: {str(e)}")
    
    def save_model_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        if not st.session_state.get('save_checkpoints', True):
            return
        
        try:
            # In a real implementation, this would save the model
            checkpoint_path = f"checkpoints/model_epoch_{epoch}.pth"
            st.info(f"Model checkpoint saved: {checkpoint_path}")
            
            # Update best model if this is the best loss so far
            best_loss = st.session_state.get('best_loss', float('inf'))
            if loss < best_loss:
                st.session_state.best_loss = loss
                st.success(f"New best model saved! Loss: {loss:.4f}")
                
        except Exception as e:
            st.error(f"Failed to save checkpoint: {str(e)}")
    
    def load_model_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            # In a real implementation, this would load the model
            st.info(f"Loading model from: {checkpoint_path}")
            
            # Update UI to reflect loaded model
            st.success("Model checkpoint loaded successfully!")
            
        except Exception as e:
            st.error(f"Failed to load checkpoint: {str(e)}")
