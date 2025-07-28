import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st

class HybridAffinityPredictor(nn.Module):
    """Hybrid Transformer-GNN model for molecular property prediction"""
    
    def __init__(self, input_dim=320, hidden_dim=256, output_dim=1, dropout=0.1):
        super(HybridAffinityPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Transformer-like attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature processing layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Property-specific heads
        self.property_heads = nn.ModuleDict({
            'binding_affinity': nn.Linear(hidden_dim, 1),
            'solubility': nn.Linear(hidden_dim, 1),
            'toxicity': nn.Linear(hidden_dim, 1),
            'stability': nn.Linear(hidden_dim, 1),
            'bioavailability': nn.Linear(hidden_dim, 1)
        })
        
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x, return_attention=False):
        """Forward pass through the hybrid model"""
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Ensure correct input dimension
        if x.shape[-1] != self.input_dim:
            # Pad or truncate to expected dimension
            if x.shape[-1] < self.input_dim:
                padding = torch.zeros(x.shape[0], self.input_dim - x.shape[-1]).to(self.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :self.input_dim]
        
        # Project to hidden dimension
        x = self.input_projection(x)
        x = self.layer_norm1(x)
        
        # Add sequence dimension for attention if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Self-attention
        residual = x
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm2(residual + attn_output)
        
        # Global pooling
        x = x.mean(dim=1)  # [batch, hidden_dim]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Main prediction
        main_output = self.mlp(x)
        
        # Property-specific predictions
        property_outputs = {}
        for prop_name, head in self.property_heads.items():
            property_outputs[prop_name] = torch.sigmoid(head(x))
        
        if return_attention:
            return main_output, property_outputs, attn_weights
        
        return main_output, property_outputs
    
    def predict(self, embedding, property_type="binding_affinity"):
        """Make predictions with the model"""
        self.eval()
        with torch.no_grad():
            try:
                output = self.forward(embedding)
                if len(output) == 3:
                    main_pred, prop_preds, _ = output
                else:
                    main_pred, prop_preds = output
                
                if property_type in prop_preds:
                    prediction = prop_preds[property_type].cpu().numpy().flatten()[0]
                else:
                    prediction = torch.sigmoid(main_pred).cpu().numpy().flatten()[0]
                
                return float(prediction)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return 0.5  # Default prediction
    
    def predict_all_properties(self, embedding):
        """Predict all molecular properties"""
        self.eval()
        with torch.no_grad():
            try:
                output = self.forward(embedding)
                if len(output) == 3:
                    main_pred, prop_preds, _ = output
                else:
                    main_pred, prop_preds = output
                
                results = {}
                for prop_name, pred_tensor in prop_preds.items():
                    results[prop_name] = float(pred_tensor.cpu().numpy().flatten()[0])
                
                # Add overall score
                results['overall_score'] = float(torch.sigmoid(main_pred).cpu().numpy().flatten()[0])
                
                return results
                
            except Exception as e:
                st.error(f"Multi-property prediction error: {str(e)}")
                return {prop: 0.5 for prop in self.property_heads.keys()}
    
    def interpret_prediction(self, embedding, property_type="binding_affinity"):
        """Get prediction with interpretation"""
        self.eval()
        with torch.no_grad():
            try:
                output = self.forward(embedding, return_attention=True)
                if len(output) == 3:
                    main_pred, prop_preds, attn_weights = output
                else:
                    main_pred, prop_preds = output
                    attn_weights = None
                
                prediction = float(prop_preds[property_type].cpu().numpy().flatten()[0])
                attention = attn_weights.cpu().numpy() if attn_weights is not None else None
                
                # Generate interpretation
                interpretation = self.generate_interpretation(prediction, property_type)
                
                return {
                    'prediction': prediction,
                    'confidence': min(0.99, max(0.01, abs(prediction - 0.5) * 2)),
                    'interpretation': interpretation,
                    'attention_weights': attention
                }
                
            except Exception as e:
                st.error(f"Interpretation error: {str(e)}")
                return {
                    'prediction': 0.5,
                    'confidence': 0.5,
                    'interpretation': "Unable to generate interpretation",
                    'attention_weights': None
                }
    
    def generate_interpretation(self, prediction, property_type):
        """Generate human-readable interpretation"""
        interpretation_map = {
            'binding_affinity': {
                'high': "Strong binding affinity predicted. This molecule shows excellent potential for target interaction.",
                'medium': "Moderate binding affinity predicted. Consider optimization for improved binding.",
                'low': "Weak binding affinity predicted. Significant modifications may be needed."
            },
            'solubility': {
                'high': "High solubility predicted. Good bioavailability characteristics expected.",
                'medium': "Moderate solubility predicted. May require formulation optimization.",
                'low': "Low solubility predicted. Poor bioavailability likely without modification."
            },
            'toxicity': {
                'high': "High toxicity risk predicted. Safety concerns need to be addressed.",
                'medium': "Moderate toxicity risk predicted. Further safety evaluation recommended.",
                'low': "Low toxicity risk predicted. Favorable safety profile expected."
            },
            'stability': {
                'high': "High stability predicted. Good shelf-life and formulation properties expected.",
                'medium': "Moderate stability predicted. May require stabilization strategies.",
                'low': "Low stability predicted. Structural modifications may be needed."
            },
            'bioavailability': {
                'high': "High bioavailability predicted. Good oral absorption expected.",
                'medium': "Moderate bioavailability predicted. May require delivery optimization.",
                'low': "Low bioavailability predicted. Alternative delivery routes may be needed."
            }
        }
        
        if property_type not in interpretation_map:
            return "Property interpretation not available."
        
        if prediction > 0.7:
            category = 'high'
        elif prediction > 0.3:
            category = 'medium'
        else:
            category = 'low'
        
        return interpretation_map[property_type][category]
    
    def get_feature_importance(self, embedding):
        """Get feature importance for interpretability"""
        # Simple gradient-based feature importance
        self.eval()
        
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding).to(self.device)
        
        embedding.requires_grad_(True)
        
        try:
            output = self.forward(embedding)
            if len(output) == 3:
                main_pred, _, _ = output
            else:
                main_pred, _ = output
            main_pred.backward()
            
            if embedding.grad is not None:
                importance = torch.abs(embedding.grad).cpu().numpy()
                return importance.flatten()
            else:
                return np.zeros(embedding.shape[-1])
            
        except Exception as e:
            st.error(f"Feature importance calculation error: {str(e)}")
            return np.zeros(embedding.shape[-1])
