import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional

class MolecularVisualizer:
    """Utility class for molecular data visualization"""
    
    def __init__(self):
        self.color_schemes = {
            'amino_acids': {
                'A': '#C8C8C8', 'R': '#145AFF', 'N': '#00DCDC', 'D': '#E60A0A',
                'C': '#E6E600', 'Q': '#00DCDC', 'E': '#E60A0A', 'G': '#EBEBEB',
                'H': '#8282D2', 'I': '#0F820F', 'L': '#0F820F', 'K': '#145AFF',
                'M': '#E6E600', 'F': '#3232AA', 'P': '#DC9682', 'S': '#FA9600',
                'T': '#FA9600', 'W': '#B45AB4', 'Y': '#3232AA', 'V': '#0F820F'
            },
            'nucleotides': {
                'A': '#FF6B6B', 'T': '#4ECDC4', 'G': '#45B7D1', 'C': '#96CEB4'
            },
            'properties': {
                'hydrophobic': '#8B4513',
                'hydrophilic': '#4169E1',
                'charged': '#FF6347',
                'polar': '#32CD32'
            }
        }
    
    def plot_sequence_composition(self, composition: Dict[str, float], seq_type: str = "protein") -> go.Figure:
        """Plot amino acid or nucleotide composition"""
        if seq_type == "protein":
            colors = [self.color_schemes['amino_acids'].get(aa, '#808080') for aa in composition.keys()]
            title = "Amino Acid Composition"
        else:
            colors = [self.color_schemes['nucleotides'].get(nt, '#808080') for nt in composition.keys()]
            title = "Nucleotide Composition"
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(composition.keys()),
                y=[v * 100 for v in composition.values()],
                marker_color=colors,
                text=[f"{v*100:.1f}%" for v in composition.values()],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Residue",
            yaxis_title="Percentage (%)",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_property_predictions(self, predictions: Dict[str, float]) -> go.Figure:
        """Plot molecular property predictions as a radar chart"""
        properties = list(predictions.keys())
        values = list(predictions.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=properties,
            fill='toself',
            name='Predicted Properties',
            line_color='rgb(59, 130, 246)',
            fillcolor='rgba(59, 130, 246, 0.2)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="Molecular Property Predictions",
            height=500
        )
        
        return fig
    
    def plot_sequence_hydrophobicity(self, sequence: str, window_size: int = 7) -> go.Figure:
        """Plot hydrophobicity along sequence"""
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        hydrophobicity_values = []
        positions = []
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            avg_hydrophobicity = np.mean([hydrophobicity_scale.get(aa, 0) for aa in window])
            hydrophobicity_values.append(avg_hydrophobicity)
            positions.append(i + window_size // 2)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=positions,
            y=hydrophobicity_values,
            mode='lines+markers',
            name='Hydrophobicity',
            line=dict(color='rgb(55, 83, 109)', width=2),
            marker=dict(size=4)
        ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig.update_layout(
            title=f"Hydrophobicity Profile (Window Size: {window_size})",
            xaxis_title="Position",
            yaxis_title="Hydrophobicity Index",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_secondary_structure(self, ss_prediction: Dict[str, float]) -> go.Figure:
        """Plot secondary structure prediction"""
        structures = list(ss_prediction.keys())
        percentages = list(ss_prediction.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=structures,
                values=percentages,
                marker_colors=colors,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>' +
                             'Percentage: %{percent}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Secondary Structure Prediction",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def plot_molecular_weight_distribution(self, sequences: List[str]) -> go.Figure:
        """Plot molecular weight distribution for multiple sequences"""
        aa_weights = {
            'A': 71.04, 'R': 156.10, 'N': 114.04, 'D': 115.03, 'C': 103.01,
            'Q': 128.06, 'E': 129.04, 'G': 57.02, 'H': 137.06, 'I': 113.08,
            'L': 113.08, 'K': 128.09, 'M': 131.04, 'F': 147.07, 'P': 97.05,
            'S': 87.03, 'T': 101.05, 'W': 186.08, 'Y': 163.06, 'V': 99.07
        }
        
        molecular_weights = []
        for seq in sequences:
            mw = sum(aa_weights.get(aa, 0) for aa in seq)
            molecular_weights.append(mw)
        
        fig = go.Figure(data=[
            go.Histogram(
                x=molecular_weights,
                nbinsx=20,
                marker_color='rgb(55, 83, 109)',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="Molecular Weight Distribution",
            xaxis_title="Molecular Weight (Da)",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_attention_heatmap(self, attention_weights: np.ndarray, sequence: str) -> go.Figure:
        """Plot attention weights as heatmap"""
        if attention_weights is None or len(attention_weights.shape) < 2:
            # Create dummy heatmap
            attention_weights = np.random.random((min(20, len(sequence)), min(20, len(sequence))))
        
        # Take first head if multi-head attention
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights[0]
        
        # Limit sequence length for visualization
        max_len = min(50, len(sequence))
        seq_truncated = sequence[:max_len]
        attn_truncated = attention_weights[:max_len, :max_len]
        
        fig = go.Figure(data=go.Heatmap(
            z=attn_truncated,
            x=list(seq_truncated),
            y=list(seq_truncated),
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Attention Weights Heatmap",
            xaxis_title="Position",
            yaxis_title="Position",
            height=500,
            width=500
        )
        
        return fig
    
    def plot_feature_importance(self, importance_scores: np.ndarray, feature_names: Optional[List[str]] = None) -> go.Figure:
        """Plot feature importance scores"""
        if feature_names is None:
            feature_names = [f"Feature {i+1}" for i in range(len(importance_scores))]
        
        # Take top 20 features for visualization
        top_indices = np.argsort(importance_scores)[-20:]
        top_scores = importance_scores[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        fig = go.Figure(data=[
            go.Bar(
                x=top_scores,
                y=top_names,
                orientation='h',
                marker_color='rgb(55, 83, 109)'
            )
        ])
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_prediction_confidence(self, predictions: List[float], confidences: List[float], labels: List[str]) -> go.Figure:
        """Plot predictions vs confidence scores"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions,
            y=confidences,
            mode='markers+text',
            text=labels,
            textposition="top center",
            marker=dict(
                size=10,
                color=confidences,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Prediction: %{x:.3f}<br>' +
                         'Confidence: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title="Prediction vs Confidence",
            xaxis_title="Prediction Score",
            yaxis_title="Confidence Score",
            height=500
        )
        
        return fig
    
    def create_3d_structure_placeholder(self, sequence: str) -> go.Figure:
        """Create a placeholder 3D structure visualization"""
        # Generate dummy 3D coordinates for visualization
        n_residues = min(len(sequence), 100)
        
        # Create a simple helix-like structure
        t = np.linspace(0, 4*np.pi, n_residues)
        x = np.cos(t)
        y = np.sin(t)
        z = t / (2*np.pi)
        
        colors = [self.color_schemes['amino_acids'].get(aa, '#808080') 
                 for aa in sequence[:n_residues]]
        
        fig = go.Figure(data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+lines',
                marker=dict(
                    size=8,
                    color=colors,
                ),
                line=dict(
                    color='gray',
                    width=4
                ),
                text=[f"{aa}{i+1}" for i, aa in enumerate(sequence[:n_residues])],
                hovertemplate='<b>%{text}</b><br>' +
                             'Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="3D Structure Visualization (Placeholder)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z"
            ),
            height=600
        )
        
        return fig
