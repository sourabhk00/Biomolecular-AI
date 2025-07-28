import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import streamlit as st

class MolecularProcessor:
    """Utility class for molecular data processing"""
    
    def __init__(self):
        self.amino_acids = set('ARNDCQEGHILKMFPSTWYV')
        self.nucleotides = set('ATCG')
        
    def validate_protein_sequence(self, sequence: str) -> Tuple[bool, str]:
        """Validate protein sequence"""
        if not sequence:
            return False, "Empty sequence"
        
        sequence = sequence.strip().upper()
        invalid_chars = set(sequence) - self.amino_acids
        
        if invalid_chars:
            return False, f"Invalid amino acids: {', '.join(invalid_chars)}"
        
        if len(sequence) < 10:
            return False, "Sequence too short (minimum 10 amino acids)"
        
        if len(sequence) > 2000:
            return False, "Sequence too long (maximum 2000 amino acids)"
        
        return True, "Valid protein sequence"
    
    def validate_dna_sequence(self, sequence: str) -> Tuple[bool, str]:
        """Validate DNA sequence"""
        if not sequence:
            return False, "Empty sequence"
        
        sequence = sequence.strip().upper()
        invalid_chars = set(sequence) - self.nucleotides
        
        if invalid_chars:
            return False, f"Invalid nucleotides: {', '.join(invalid_chars)}"
        
        if len(sequence) < 20:
            return False, "Sequence too short (minimum 20 nucleotides)"
        
        return True, "Valid DNA sequence"
    
    def parse_fasta(self, fasta_content: str) -> Dict[str, str]:
        """Parse FASTA format content"""
        sequences = {}
        current_name = None
        current_seq = []
        
        for line in fasta_content.split('\n'):
            line = line.strip()
            if line.startswith('>'):
                if current_name and current_seq:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)
        
        # Add the last sequence
        if current_name and current_seq:
            sequences[current_name] = ''.join(current_seq)
        
        return sequences
    
    def generate_fasta(self, sequences: Dict[str, str]) -> str:
        """Generate FASTA format content"""
        fasta_content = []
        for name, seq in sequences.items():
            fasta_content.append(f">{name}")
            # Break sequence into lines of 80 characters
            for i in range(0, len(seq), 80):
                fasta_content.append(seq[i:i+80])
        
        return '\n'.join(fasta_content)
    
    def translate_dna_to_protein(self, dna_sequence: str) -> str:
        """Translate DNA sequence to protein"""
        genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }
        
        dna_sequence = dna_sequence.upper().strip()
        protein_sequence = []
        
        for i in range(0, len(dna_sequence) - 2, 3):
            codon = dna_sequence[i:i+3]
            if len(codon) == 3:
                amino_acid = genetic_code.get(codon, 'X')
                if amino_acid == '*':  # Stop codon
                    break
                protein_sequence.append(amino_acid)
        
        return ''.join(protein_sequence)
    
    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Calculate sequence similarity using simple alignment"""
        if not seq1 or not seq2:
            return 0.0
        
        # Simple alignment score
        min_length = min(len(seq1), len(seq2))
        max_length = max(len(seq1), len(seq2))
        
        matches = sum(1 for i in range(min_length) if seq1[i] == seq2[i])
        similarity = matches / max_length
        
        return similarity
    
    def generate_consensus_sequence(self, sequences: List[str]) -> str:
        """Generate consensus sequence from multiple sequences"""
        if not sequences:
            return ""
        
        # Find the minimum length
        min_length = min(len(seq) for seq in sequences)
        consensus = []
        
        for i in range(min_length):
            # Count amino acids at each position
            aa_counts = {}
            for seq in sequences:
                aa = seq[i]
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            # Find the most common amino acid
            if aa_counts:
                consensus_aa = max(aa_counts, key=lambda x: aa_counts[x])
                consensus.append(consensus_aa)
        
        return ''.join(consensus)
    
    def detect_sequence_type(self, sequence: str) -> str:
        """Auto-detect sequence type (DNA or protein)"""
        sequence = sequence.strip().upper()
        
        if not sequence:
            return "unknown"
        
        # Count nucleotide characters
        nucleotide_count = sum(1 for char in sequence if char in 'ATCG')
        nucleotide_ratio = nucleotide_count / len(sequence)
        
        if nucleotide_ratio > 0.95:
            return "dna"
        elif nucleotide_ratio < 0.1:
            return "protein"
        else:
            return "unknown"
    
    def extract_sequence_features(self, sequence: str, seq_type: str = None) -> Dict:
        """Extract basic sequence features"""
        if not sequence:
            return {}
        
        sequence = sequence.strip().upper()
        
        if seq_type is None:
            seq_type = self.detect_sequence_type(sequence)
        
        features = {
            'length': len(sequence),
            'type': seq_type,
            'gc_content': None,
            'composition': {}
        }
        
        if seq_type == "dna":
            gc_count = sequence.count('G') + sequence.count('C')
            features['gc_content'] = gc_count / len(sequence) if len(sequence) > 0 else 0
            
            for nucleotide in 'ATCG':
                features['composition'][nucleotide] = sequence.count(nucleotide) / len(sequence)
        
        elif seq_type == "protein":
            for aa in self.amino_acids:
                features['composition'][aa] = sequence.count(aa) / len(sequence)
        
        return features
    
    def find_motifs(self, sequence: str, motif_patterns: List[str]) -> Dict[str, List[int]]:
        """Find motif patterns in sequence"""
        sequence = sequence.upper()
        motif_positions = {}
        
        for pattern in motif_patterns:
            pattern = pattern.upper()
            positions = []
            start = 0
            
            while True:
                pos = sequence.find(pattern, start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            
            motif_positions[pattern] = positions
        
        return motif_positions
    
    def reverse_complement(self, dna_sequence: str) -> str:
        """Generate reverse complement of DNA sequence"""
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        
        dna_sequence = dna_sequence.upper()
        complement = ''.join(complement_map.get(base, base) for base in dna_sequence)
        
        return complement[::-1]  # Reverse the string
    
    def calculate_melting_temperature(self, dna_sequence: str) -> float:
        """Calculate approximate DNA melting temperature"""
        # Simple approximation: Tm = 64.9 + 41 * (G+C-16.4) / length
        if not dna_sequence:
            return 0.0
        
        dna_sequence = dna_sequence.upper()
        gc_content = (dna_sequence.count('G') + dna_sequence.count('C')) / len(dna_sequence)
        
        # Wallace rule for short sequences
        if len(dna_sequence) <= 14:
            tm = (dna_sequence.count('A') + dna_sequence.count('T')) * 2 + \
                 (dna_sequence.count('G') + dna_sequence.count('C')) * 4
        else:
            tm = 64.9 + 41 * (gc_content - 0.164)
        
        return round(tm, 1)
