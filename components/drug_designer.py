import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class DrugDesignerComponent:
    """Streamlit component for AI-powered drug design"""
    
    def __init__(self, hybrid_predictor, molecular_processor, visualizer):
        self.hybrid_predictor = hybrid_predictor
        self.molecular_processor = molecular_processor
        self.visualizer = visualizer
    
    def render(self):
        """Render the drug design interface"""
        # Input section
        self.render_input_section()
        
        # Design section
        if self.has_valid_inputs():
            self.render_design_section()
    
    def render_input_section(self):
        """Render drug design input section"""
        st.subheader("Drug Design Parameters")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Target Protein Information**")
            
            target_sequence = st.text_area(
                "Target protein sequence:",
                height=100,
                placeholder="Enter the protein sequence of your drug target...",
                key="target_sequence"
            )
            
            target_type = st.selectbox(
                "Target type:",
                ["Enzyme", "Receptor", "Ion Channel", "Transporter", "Other"],
                key="target_type"
            )
            
            binding_site = st.text_input(
                "Known binding site (optional):",
                placeholder="e.g., ATP binding site, allosteric site",
                key="binding_site"
            )
        
        with col2:
            st.write("**Drug Design Preferences**")
            
            drug_type = st.selectbox(
                "Drug type:",
                ["Small Molecule", "Peptide", "Antibody", "Nucleic Acid"],
                key="drug_type"
            )
            
            administration_route = st.selectbox(
                "Preferred administration route:",
                ["Oral", "Intravenous", "Topical", "Inhalation", "Subcutaneous"],
                key="administration_route"
            )
            
            molecular_weight_range = st.slider(
                "Molecular weight range (Da):",
                min_value=100,
                max_value=2000,
                value=(300, 800),
                key="mw_range"
            )
    
    def render_design_section(self):
        """Render drug design results and analysis"""
        st.subheader("🎯 AI Drug Design Results")
        
        target_sequence = st.session_state.get("target_sequence", "")
        
        # Generate drug candidates
        if st.button("Generate Drug Candidates", type="primary"):
            with st.spinner("Designing drug candidates..."):
                candidates = self.generate_drug_candidates(target_sequence)
                st.session_state.drug_candidates = candidates
                st.rerun()
        
        # Display candidates if available
        if 'drug_candidates' in st.session_state:
            self.render_candidate_results()
    
    def has_valid_inputs(self) -> bool:
        """Check if we have valid inputs for drug design"""
        target_sequence = st.session_state.get("target_sequence", "")
        return bool(target_sequence and len(target_sequence.strip()) > 10)
    
    def generate_drug_candidates(self, target_sequence: str) -> List[Dict]:
        """Generate drug candidates using AI models"""
        # Validate target sequence
        is_valid, message = self.molecular_processor.validate_protein_sequence(target_sequence)
        if not is_valid:
            st.error(f"Invalid target sequence: {message}")
            return []
        
        # Generate embedding for target
        from models.esm_model import ESMProteinAnalyzer
        esm_analyzer = ESMProteinAnalyzer()
        target_embedding = esm_analyzer.get_embedding(target_sequence)
        
        # Generate multiple drug candidates
        candidates = []
        drug_types = ["small_molecule", "peptide", "modified_peptide"]
        
        for i in range(5):  # Generate 5 candidates
            candidate = self.generate_single_candidate(target_embedding, i, drug_types[i % len(drug_types)])
            candidates.append(candidate)
        
        return candidates
    
    def generate_single_candidate(self, target_embedding: np.ndarray, candidate_id: int, drug_type: str) -> Dict:
        """Generate a single drug candidate"""
        # Predict molecular properties
        properties = self.hybrid_predictor.predict_all_properties(target_embedding)
        
        # Generate candidate structure based on type
        if drug_type == "small_molecule":
            candidate = self.generate_small_molecule_candidate(candidate_id, properties)
        elif drug_type == "peptide":
            candidate = self.generate_peptide_candidate(candidate_id, properties)
        else:
            candidate = self.generate_modified_peptide_candidate(candidate_id, properties)
        
        # Add AI predictions
        candidate.update({
            'predicted_properties': properties,
            'design_confidence': np.random.uniform(0.7, 0.95),  # Placeholder
            'novelty_score': np.random.uniform(0.6, 0.9),  # Placeholder
        })
        
        return candidate
    
    def generate_small_molecule_candidate(self, candidate_id: int, properties: Dict) -> Dict:
        """Generate a small molecule drug candidate"""
        # Placeholder small molecule structures
        smiles_examples = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen-like
            "CC1=C(C=C(C=C1)C(=O)NC2=CC=CC=C2)OC",  # Modified aspirin-like
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine-like
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",  # Albuterol-like
            "CN(C)C1=NC=NC2=C1C=CN2[C@H]3[C@@H]([C@@H]([C@H](O3)CO)O)O"  # Modified nucleoside
        ]
        
        return {
            'candidate_id': f"SM_{candidate_id + 1:03d}",
            'type': 'Small Molecule',
            'smiles': smiles_examples[candidate_id % len(smiles_examples)],
            'molecular_weight': np.random.uniform(200, 600),
            'logp': np.random.uniform(1, 4),
            'tpsa': np.random.uniform(20, 120),
            'hbd': np.random.randint(0, 5),
            'hba': np.random.randint(1, 8),
            'rotatable_bonds': np.random.randint(1, 10)
        }
    
    def generate_peptide_candidate(self, candidate_id: int, properties: Dict) -> Dict:
        """Generate a peptide drug candidate"""
        # Generate random peptide sequences
        amino_acids = list('ARNDCQEGHILKMFPSTWYV')
        peptide_length = np.random.randint(5, 15)
        sequence = ''.join(np.random.choice(amino_acids, peptide_length))
        
        return {
            'candidate_id': f"PEP_{candidate_id + 1:03d}",
            'type': 'Peptide',
            'sequence': sequence,
            'length': peptide_length,
            'molecular_weight': peptide_length * 110,  # Average MW per AA
            'net_charge': np.random.randint(-3, 4),
            'hydrophobicity': np.random.uniform(-2, 2)
        }
    
    def generate_modified_peptide_candidate(self, candidate_id: int, properties: Dict) -> Dict:
        """Generate a modified peptide drug candidate"""
        # Generate peptide with modifications
        amino_acids = list('ARNDCQEGHILKMFPSTWYV')
        peptide_length = np.random.randint(8, 20)
        sequence = ''.join(np.random.choice(amino_acids, peptide_length))
        
        modifications = ["N-methylation", "Cyclization", "D-amino acids", "PEGylation"]
        selected_mod = np.random.choice(modifications)
        
        return {
            'candidate_id': f"MOD_{candidate_id + 1:03d}",
            'type': 'Modified Peptide',
            'sequence': sequence,
            'modification': selected_mod,
            'length': peptide_length,
            'molecular_weight': peptide_length * 120,  # Modified MW
            'stability_score': np.random.uniform(0.7, 0.95)
        }
    
    def render_candidate_results(self):
        """Render drug candidate results"""
        candidates = st.session_state.drug_candidates
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Candidates Generated", len(candidates))
        
        with col2:
            avg_confidence = np.mean([c['design_confidence'] for c in candidates])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col3:
            avg_novelty = np.mean([c['novelty_score'] for c in candidates])
            st.metric("Avg Novelty", f"{avg_novelty:.2f}")
        
        with col4:
            high_potential = sum(1 for c in candidates if c['design_confidence'] > 0.8)
            st.metric("High Potential", high_potential)
        
        # Candidate details
        st.subheader("💊 Drug Candidates")
        
        for i, candidate in enumerate(candidates):
            with st.expander(f"Candidate {candidate['candidate_id']} - {candidate['type']}", expanded=(i == 0)):
                self.render_single_candidate(candidate)
        
        # Comparative analysis
        self.render_comparative_analysis(candidates)
    
    def render_single_candidate(self, candidate: Dict):
        """Render details for a single drug candidate"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.write("**Basic Properties**")
            st.write(f"Type: {candidate['type']}")
            st.write(f"ID: {candidate['candidate_id']}")
            
            if 'molecular_weight' in candidate:
                st.write(f"MW: {candidate['molecular_weight']:.1f} Da")
            
            if 'sequence' in candidate:
                st.write(f"Sequence: `{candidate['sequence']}`")
            
            if 'smiles' in candidate:
                st.write(f"SMILES: `{candidate['smiles']}`")
        
        with col2:
            st.write("**AI Predictions**")
            properties = candidate['predicted_properties']
            
            for prop_name, value in properties.items():
                if prop_name != 'overall_score':
                    st.metric(prop_name.replace('_', ' ').title(), f"{value:.3f}")
        
        with col3:
            st.write("**Design Metrics**")
            st.metric("Confidence", f"{candidate['design_confidence']:.3f}")
            st.metric("Novelty", f"{candidate['novelty_score']:.3f}")
            
            # Overall assessment
            overall_score = candidate['predicted_properties']['overall_score']
            if overall_score > 0.7:
                st.success("High Potential 🌟")
            elif overall_score > 0.5:
                st.warning("Moderate Potential ⚠️")
            else:
                st.error("Low Potential ❌")
        
        # Property radar chart
        properties = candidate['predicted_properties']
        fig = self.visualizer.plot_property_predictions(properties)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_comparative_analysis(self, candidates: List[Dict]):
        """Render comparative analysis of candidates"""
        st.subheader("📊 Comparative Analysis")
        
        # Create comparison dataframe
        comparison_data = []
        for candidate in candidates:
            row = {
                'ID': candidate['candidate_id'],
                'Type': candidate['type'],
                'Confidence': candidate['design_confidence'],
                'Novelty': candidate['novelty_score'],
                'Overall Score': candidate['predicted_properties']['overall_score']
            }
            
            # Add molecular properties
            for prop, value in candidate['predicted_properties'].items():
                if prop != 'overall_score':
                    row[prop.replace('_', ' ').title()] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        st.write("**Candidate Comparison Table**")
        st.dataframe(df, use_container_width=True)
        
        # Plot comparisons
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence vs Novelty scatter plot
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            for i, candidate in enumerate(candidates):
                fig.add_trace(go.Scatter(
                    x=[candidate['design_confidence']],
                    y=[candidate['novelty_score']],
                    mode='markers+text',
                    text=[candidate['candidate_id']],
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=candidate['predicted_properties']['overall_score'],
                        colorscale='Viridis',
                        showscale=(i == 0)
                    ),
                    name=candidate['candidate_id']
                ))
            
            fig.update_layout(
                title="Design Confidence vs Novelty",
                xaxis_title="Design Confidence",
                yaxis_title="Novelty Score",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Property comparison radar
            if len(candidates) > 0:
                properties = candidates[0]['predicted_properties']
                property_names = [k for k in properties.keys() if k != 'overall_score']
                
                fig = go.Figure()
                
                for candidate in candidates[:3]:  # Show top 3 for clarity
                    values = [candidate['predicted_properties'][prop] for prop in property_names]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=property_names,
                        fill='toself',
                        name=candidate['candidate_id']
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title="Property Comparison (Top 3)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Candidate Analysis",
            data=csv,
            file_name="drug_candidates.csv",
            mime="text/csv"
        )
