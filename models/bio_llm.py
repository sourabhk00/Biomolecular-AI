import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import os
import json


class BiomolecularLLM(nn.Module):
    """Large Language Model for biomolecular sequence generation and analysis"""
    
    def __init__(self, vocab_size=25, embed_dim=512, num_heads=8, num_layers=12, max_seq_len=2048):
        super(BiomolecularLLM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Amino acid vocabulary (20 standard + special tokens)
        self.vocab = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19,
            '<PAD>': 20, '<START>': 21, '<END>': 22, '<UNK>': 23, '<MASK>': 24
        }
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Output heads
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Property prediction heads
        self.property_heads = nn.ModuleDict({
            'stability': nn.Linear(embed_dim, 1),
            'solubility': nn.Linear(embed_dim, 1),
            'binding_affinity': nn.Linear(embed_dim, 1),
            'toxicity': nn.Linear(embed_dim, 1),
            'activity': nn.Linear(embed_dim, 1)
        })
        
        self.dropout = nn.Dropout(0.1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the biomolecular LLM"""
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.output_projection(hidden_states)
        
        # Property prediction heads
        pooled_output = hidden_states.mean(dim=1)  # Global average pooling
        property_predictions = {}
        for prop_name, head in self.property_heads.items():
            property_predictions[prop_name] = torch.sigmoid(head(pooled_output))
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
            'property_predictions': property_predictions
        }
        
        if labels is not None:
            # Calculate language modeling loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.vocab['<PAD>'])
            lm_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            outputs['loss'] = lm_loss
        
        return outputs
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode protein sequence to token IDs"""
        tokens = ['<START>'] + list(sequence.upper()) + ['<END>']
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        return torch.tensor(token_ids, dtype=torch.long)
    
    def decode_sequence(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to protein sequence"""
        tokens = [self.reverse_vocab.get(id.item(), '<UNK>') for id in token_ids]
        # Remove special tokens
        sequence = ''.join([t for t in tokens if t not in ['<START>', '<END>', '<PAD>', '<UNK>']])
        return sequence
    
    def generate_sequence(self, prompt: str = "", max_length: int = 100, temperature: float = 1.0) -> str:
        """Generate new protein sequence using the LLM"""
        self.eval()
        
        if prompt:
            input_ids = self.encode_sequence(prompt)
        else:
            input_ids = torch.tensor([self.vocab['<START>']], dtype=torch.long)
        
        input_ids = input_ids.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if next_token.item() == self.vocab['<END>']:
                    break
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        generated_sequence = self.decode_sequence(input_ids.squeeze(0))
        return generated_sequence
    
    def predict_properties(self, sequence: str) -> Dict[str, float]:
        """Predict molecular properties for a given sequence"""
        self.eval()
        
        input_ids = self.encode_sequence(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids)
            predictions = outputs['property_predictions']
        
        return {prop: pred.cpu().item() for prop, pred in predictions.items()}
    
    def get_attention_weights(self, sequence: str) -> np.ndarray:
        """Get attention weights for interpretability"""
        # Simplified attention extraction for visualization
        self.eval()
        input_ids = self.encode_sequence(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(input_ids)
            # Return dummy attention for now - in real implementation would extract from transformer layers
            seq_len = input_ids.shape[1]
            attention = torch.randn(seq_len, seq_len)
            return F.softmax(attention, dim=-1).numpy()


class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=attention_mask)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        return x


class DiffusionMolecularGenerator(nn.Module):
    """Diffusion model for molecular structure generation"""
    
    def __init__(self, feature_dim: int = 256, num_timesteps: int = 1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_timesteps = num_timesteps
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(feature_dim + 1, 512),  # +1 for timestep embedding
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Predict noise to be removed from noisy input"""
        # Simple timestep embedding
        t_embed = timestep.float().unsqueeze(-1) / self.num_timesteps
        
        # Concatenate input with timestep
        x_t = torch.cat([x, t_embed.expand(-1, x.shape[1]).unsqueeze(-1)], dim=-1)
        
        return self.noise_predictor(x_t)
    
    def generate_molecule(self, num_atoms: int = 50) -> torch.Tensor:
        """Generate molecular coordinates using diffusion process"""
        # Start with random noise
        x = torch.randn(1, num_atoms, self.feature_dim).to(self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            timestep = torch.tensor([t]).to(self.device)
            
            with torch.no_grad():
                predicted_noise = self.forward(x, timestep)
                
                # Remove predicted noise (simplified)
                alpha = 1.0 - t / self.num_timesteps
                x = x * alpha + predicted_noise * (1 - alpha) * 0.1
        
        return x


class ProteinFoldingPredictor(nn.Module):
    """Deep neural network for protein structure prediction"""
    
    def __init__(self, sequence_dim: int = 20, hidden_dim: int = 512):
        super().__init__()
        
        # Sequence processing
        self.sequence_encoder = nn.Sequential(
            nn.Linear(sequence_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Structure prediction heads
        self.secondary_structure_head = nn.Linear(hidden_dim, 3)  # Helix, Sheet, Coil
        self.phi_psi_head = nn.Linear(hidden_dim, 2)  # Dihedral angles
        self.contact_head = nn.Linear(hidden_dim * 2, 1)  # Contact prediction
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def forward(self, sequence_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict protein structure features"""
        encoded = self.sequence_encoder(sequence_features)
        
        # Secondary structure prediction
        ss_logits = self.secondary_structure_head(encoded)
        
        # Dihedral angle prediction
        angles = self.phi_psi_head(encoded)
        
        # Contact map prediction (simplified)
        seq_len = encoded.shape[1]
        contact_features = torch.zeros(encoded.shape[0], seq_len, seq_len, encoded.shape[2] * 2).to(self.device)
        for i in range(seq_len):
            for j in range(seq_len):
                contact_features[:, i, j, :] = torch.cat([encoded[:, i, :], encoded[:, j, :]], dim=-1)
        
        contact_logits = self.contact_head(contact_features).squeeze(-1)
        
        return {
            'secondary_structure': F.softmax(ss_logits, dim=-1),
            'dihedral_angles': angles,
            'contact_map': torch.sigmoid(contact_logits)
        }


class QuantumMolecularOracle(nn.Module):
    """Quantum-inspired molecular property predictor"""
    
    def __init__(self, input_dim: int = 256, num_qubits: int = 8):
        super().__init__()
        self.num_qubits = num_qubits
        
        # Classical preprocessing
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Quantum-inspired layers (simulated)
        self.quantum_weights = nn.Parameter(torch.randn(num_qubits, num_qubits))
        self.quantum_bias = nn.Parameter(torch.randn(num_qubits))
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(num_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def quantum_layer(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate quantum computation for molecular properties"""
        # Simplified quantum-inspired transformation
        quantum_state = torch.mm(x, self.quantum_weights) + self.quantum_bias
        
        # Apply quantum-inspired activation (rotation gates simulation)
        quantum_state = torch.cos(quantum_state) + 1j * torch.sin(quantum_state)
        quantum_state = torch.real(quantum_state * torch.conj(quantum_state))
        
        return quantum_state
    
    def forward(self, molecular_features: torch.Tensor) -> torch.Tensor:
        """Predict molecular properties using quantum-inspired computation"""
        features = self.feature_processor(molecular_features)
        
        # Pad or truncate to match quantum layer input
        if features.shape[-1] < self.num_qubits:
            padding = torch.zeros(features.shape[0], self.num_qubits - features.shape[-1]).to(self.device)
            features = torch.cat([features, padding], dim=-1)
        else:
            features = features[:, :self.num_qubits]
        
        quantum_output = self.quantum_layer(features)
        prediction = self.output_layer(quantum_output)
        
        return torch.sigmoid(prediction)
