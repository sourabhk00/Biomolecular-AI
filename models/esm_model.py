import torch
import torch.nn as nn
import numpy as np
import streamlit as st
import os


# Note: For full ESM-2 integration, uncomment the line below and install transformers
# from transformers import AutoTokenizer, AutoModel

class ESMProteinAnalyzer:
    """ESM-2 based protein analyzer for embeddings and property prediction"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """Load ESM-2 model and tokenizer"""
        try:
            # For demonstration, we'll use the fallback model
            # To enable ESM-2: install transformers and uncomment the lines below
            
            # hf_token = os.getenv("HF_TOKEN", None)
            # self.tokenizer = AutoTokenizer.from_pretrained(
            #     self.model_name, 
            #     do_lower_case=False,
            #     token=hf_token
            # )
            # self.model = AutoModel.from_pretrained(
            #     self.model_name,
            #     token=hf_token
            # )
            # self.model.to(self.device)
            # self.model.eval()
            
            # Using fallback for demo
            st.warning("Using fallback protein embedding model. Install 'transformers' for ESM-2 integration.")
            self.use_fallback_model()
            
        except Exception as e:
            st.error(f"Error loading ESM model: {str(e)}")
            self.use_fallback_model()
    
    def use_fallback_model(self):
        """Use a simple fallback model if ESM-2 fails to load"""
        self.model = None
        self.tokenizer = None
    
    def get_embedding(self, sequence, max_length=512):
        """Get protein sequence embedding using ESM-2"""
        if self.model is None:
            # Fallback embedding
            return self.get_fallback_embedding(sequence)
        
        try:
            # Truncate sequence if too long
            if len(sequence) > max_length - 2:
                sequence = sequence[:max_length-2]
            
            # This would be used with real ESM-2 model
            # inputs = self.tokenizer(sequence, return_tensors="pt", ...)
            # For now, fallback to simple embedding
            return self.get_fallback_embedding(sequence)
            
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            return self.get_fallback_embedding(sequence)
    
    def get_fallback_embedding(self, sequence):
        """Generate a simple fallback embedding"""
        # Create a simple amino acid frequency-based embedding
        aa_dict = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
            'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
        # Create frequency vector
        freq_vector = np.zeros(20)
        valid_sequence = ''.join([aa for aa in sequence.upper() if aa in aa_dict])
        
        if len(valid_sequence) > 0:
            for aa in valid_sequence:
                if aa in aa_dict:
                    freq_vector[aa_dict[aa]] += 1
            freq_vector = freq_vector / len(valid_sequence)
        
        # Pad to expected dimension
        embedding = np.pad(freq_vector, (0, 320 - len(freq_vector)), 'constant')
        return embedding.reshape(1, -1)
    
    def analyze_sequence(self, sequence):
        """Comprehensive sequence analysis"""
        if not sequence or len(sequence.strip()) == 0:
            return None
        
        sequence = sequence.strip().upper()
        
        # Basic sequence properties
        analysis = {
            'length': len(sequence),
            'molecular_weight': self.calculate_molecular_weight(sequence),
            'isoelectric_point': self.calculate_isoelectric_point(sequence),
            'hydrophobicity': self.calculate_hydrophobicity(sequence),
            'amino_acid_composition': self.get_amino_acid_composition(sequence),
            'secondary_structure_prediction': self.predict_secondary_structure(sequence)
        }
        
        # Get embedding
        embedding = self.get_embedding(sequence)
        analysis['embedding'] = embedding
        analysis['embedding_dim'] = embedding.shape[1] if len(embedding.shape) > 1 else len(embedding)
        
        return analysis
    
    def calculate_molecular_weight(self, sequence):
        """Calculate approximate molecular weight"""
        aa_weights = {
            'A': 71.04, 'R': 156.10, 'N': 114.04, 'D': 115.03, 'C': 103.01,
            'Q': 128.06, 'E': 129.04, 'G': 57.02, 'H': 137.06, 'I': 113.08,
            'L': 113.08, 'K': 128.09, 'M': 131.04, 'F': 147.07, 'P': 97.05,
            'S': 87.03, 'T': 101.05, 'W': 186.08, 'Y': 163.06, 'V': 99.07
        }
        
        weight = sum(aa_weights.get(aa, 0) for aa in sequence)
        return round(weight, 2)
    
    def calculate_isoelectric_point(self, sequence):
        """Calculate approximate isoelectric point"""
        # Simplified calculation
        positive_residues = sequence.count('R') + sequence.count('K') + sequence.count('H')
        negative_residues = sequence.count('D') + sequence.count('E')
        
        if len(sequence) == 0:
            return 7.0
        
        charge_ratio = (positive_residues - negative_residues) / len(sequence)
        pi = 7.0 + charge_ratio * 3.0  # Simplified approximation
        return round(max(1.0, min(14.0, pi)), 2)
    
    def calculate_hydrophobicity(self, sequence):
        """Calculate hydrophobicity index"""
        hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        if len(sequence) == 0:
            return 0.0
        
        hydrophobicity = sum(hydrophobicity_scale.get(aa, 0) for aa in sequence) / len(sequence)
        return round(hydrophobicity, 3)
    
    def get_amino_acid_composition(self, sequence):
        """Get amino acid composition"""
        aa_count = {}
        for aa in 'ARNDCQEGHILKMFPSTWYV':
            aa_count[aa] = sequence.count(aa)
        
        total = len(sequence)
        if total == 0:
            return {aa: 0.0 for aa in aa_count}
        
        return {aa: round(count/total * 100, 2) for aa, count in aa_count.items()}
    
    def predict_secondary_structure(self, sequence):
        """Simple secondary structure prediction"""
        # Very simplified rules-based prediction
        helix_prone = 'AEHKQR'
        sheet_prone = 'FILVWY'
        
        if len(sequence) == 0:
            return {'helix': 0, 'sheet': 0, 'coil': 0}
        
        helix_score = sum(1 for aa in sequence if aa in helix_prone) / len(sequence)
        sheet_score = sum(1 for aa in sequence if aa in sheet_prone) / len(sequence)
        coil_score = 1 - helix_score - sheet_score
        
        return {
            'helix': round(helix_score * 100, 1),
            'sheet': round(sheet_score * 100, 1),
            'coil': round(max(0, coil_score * 100), 1)
        }
