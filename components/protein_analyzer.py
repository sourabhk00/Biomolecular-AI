import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional

class ProteinAnalyzerComponent:
    """Streamlit component for protein sequence analysis"""
    
    def __init__(self, esm_analyzer, visualizer):
        self.esm_analyzer = esm_analyzer
        self.visualizer = visualizer
    
    def render(self):
        """Render the protein analyzer interface"""
        # Input section
        self.render_input_section()
        
        # Analysis section
        if 'protein_sequence' in st.session_state and st.session_state.protein_sequence:
            self.render_analysis_section()
    
    def render_input_section(self):
        """Render protein sequence input section"""
        st.subheader("Protein Sequence Input")
        
        # Tabs for different input methods
        tab1, tab2, tab3 = st.tabs(["Manual Entry", "File Upload", "Example Sequences"])
        
        with tab1:
            protein_sequence = st.text_area(
                "Enter protein sequence:",
                height=150,
                placeholder="MKFLVNVALVFMVVYISYIYAA...",
                help="Enter a single-letter amino acid sequence"
            )
            
            if st.button("Analyze Sequence", type="primary"):
                if protein_sequence.strip():
                    is_valid, message = self.esm_analyzer.molecular_processor.validate_protein_sequence(protein_sequence)
                    if is_valid:
                        st.session_state.protein_sequence = protein_sequence.strip().upper()
                        st.session_state.analysis_results = None
                        st.rerun()
                    else:
                        st.error(f"Invalid sequence: {message}")
                else:
                    st.error("Please enter a protein sequence")
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Upload FASTA file",
                type=['fasta', 'fa', 'txt'],
                help="Upload a FASTA file containing protein sequences"
            )
            
            if uploaded_file is not None:
                content = str(uploaded_file.read(), "utf-8")
                sequences = self.esm_analyzer.molecular_processor.parse_fasta(content)
                
                if sequences:
                    selected_sequence = st.selectbox(
                        "Select sequence to analyze:",
                        list(sequences.keys())
                    )
                    
                    if st.button("Analyze Selected Sequence"):
                        protein_sequence = sequences[selected_sequence]
                        is_valid, message = self.esm_analyzer.molecular_processor.validate_protein_sequence(protein_sequence)
                        if is_valid:
                            st.session_state.protein_sequence = protein_sequence
                            st.session_state.sequence_name = selected_sequence
                            st.session_state.analysis_results = None
                            st.rerun()
                        else:
                            st.error(f"Invalid sequence: {message}")
                else:
                    st.error("No valid sequences found in the uploaded file")
        
        with tab3:
            example_sequences = {
                "Human Insulin": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
                "Green Fluorescent Protein": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                "Lysozyme": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
            }
            
            selected_example = st.selectbox(
                "Choose an example protein:",
                list(example_sequences.keys())
            )
            
            st.text_area(
                "Example sequence:",
                value=example_sequences[selected_example],
                height=100,
                disabled=True
            )
            
            if st.button("Analyze Example"):
                st.session_state.protein_sequence = example_sequences[selected_example]
                st.session_state.sequence_name = selected_example
                st.session_state.analysis_results = None
                st.rerun()
    
    def render_analysis_section(self):
        """Render protein analysis results"""
        sequence = st.session_state.protein_sequence
        sequence_name = getattr(st.session_state, 'sequence_name', 'Unknown')
        
        st.subheader(f"Analysis Results: {sequence_name}")
        
        # Perform analysis if not cached
        if st.session_state.get('analysis_results') is None:
            with st.spinner("Analyzing protein sequence..."):
                analysis = self.esm_analyzer.analyze_sequence(sequence)
                st.session_state.analysis_results = analysis
        
        analysis = st.session_state.analysis_results
        
        if analysis is None:
            st.error("Failed to analyze sequence")
            return
        
        # Display basic properties
        self.render_basic_properties(analysis)
        
        # Display composition analysis
        self.render_composition_analysis(analysis)
        
        # Display secondary structure prediction
        self.render_secondary_structure(analysis)
        
        # Display hydrophobicity profile
        self.render_hydrophobicity_profile(sequence)
        
        # Display embedding visualization
        self.render_embedding_analysis(analysis)
        
        # Display 3D structure placeholder
        self.render_3d_structure(sequence)
    
    def render_basic_properties(self, analysis: Dict):
        """Render basic protein properties"""
        st.subheader("📊 Basic Properties")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Length", f"{analysis['length']} aa")
        
        with col2:
            st.metric("Molecular Weight", f"{analysis['molecular_weight']:.1f} Da")
        
        with col3:
            st.metric("Isoelectric Point", f"{analysis['isoelectric_point']:.2f}")
        
        with col4:
            st.metric("Hydrophobicity", f"{analysis['hydrophobicity']:.3f}")
    
    def render_composition_analysis(self, analysis: Dict):
        """Render amino acid composition analysis"""
        st.subheader("🧬 Amino Acid Composition")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot composition
            fig = self.visualizer.plot_sequence_composition(
                analysis['amino_acid_composition'], 
                seq_type="protein"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Show top amino acids
            composition = analysis['amino_acid_composition']
            sorted_composition = sorted(composition.items(), key=lambda x: x[1], reverse=True)
            
            st.write("**Most Abundant:**")
            for aa, percent in sorted_composition[:5]:
                st.write(f"{aa}: {percent:.1f}%")
            
            # Calculate hydrophobic vs hydrophilic ratio
            hydrophobic_aas = set('AILMFPWV')
            hydrophilic_aas = set('RNDQEHKST')
            
            hydrophobic_percent = sum(composition[aa] for aa in hydrophobic_aas if aa in composition)
            hydrophilic_percent = sum(composition[aa] for aa in hydrophilic_aas if aa in composition)
            
            st.metric("Hydrophobic %", f"{hydrophobic_percent:.1f}")
            st.metric("Hydrophilic %", f"{hydrophilic_percent:.1f}")
    
    def render_secondary_structure(self, analysis: Dict):
        """Render secondary structure prediction"""
        st.subheader("🔄 Secondary Structure Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            ss_prediction = analysis['secondary_structure_prediction']
            fig = self.visualizer.plot_secondary_structure(ss_prediction)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Predicted Distribution:**")
            for structure, percentage in analysis['secondary_structure_prediction'].items():
                st.metric(structure.title(), f"{percentage}%")
            
            st.info(
                "**Structure Types:**\n"
                "- **Helix**: α-helical regions\n"
                "- **Sheet**: β-sheet regions\n"
                "- **Coil**: Random coil/loop regions"
            )
    
    def render_hydrophobicity_profile(self, sequence: str):
        """Render hydrophobicity profile"""
        st.subheader("💧 Hydrophobicity Profile")
        
        window_size = st.slider(
            "Window size for smoothing:",
            min_value=3,
            max_value=21,
            value=7,
            step=2,
            help="Size of the sliding window for calculating average hydrophobicity"
        )
        
        fig = self.visualizer.plot_sequence_hydrophobicity(sequence, window_size)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(
            "**Interpretation:**\n"
            "- Positive values indicate hydrophobic regions\n"
            "- Negative values indicate hydrophilic regions\n"
            "- Peaks may indicate transmembrane domains or binding sites"
        )
    
    def render_embedding_analysis(self, analysis: Dict):
        """Render embedding analysis"""
        st.subheader("🔬 ESM-2 Embedding Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            embedding = analysis['embedding']
            if embedding is not None:
                st.metric("Embedding Dimension", analysis['embedding_dim'])
                
                # Show embedding statistics
                embedding_flat = embedding.flatten() if len(embedding.shape) > 1 else embedding
                st.metric("Mean Activation", f"{np.mean(embedding_flat):.4f}")
                st.metric("Std Activation", f"{np.std(embedding_flat):.4f}")
                st.metric("Max Activation", f"{np.max(embedding_flat):.4f}")
            else:
                st.error("Failed to generate embeddings")
        
        with col2:
            if embedding is not None:
                # Plot embedding distribution
                embedding_flat = embedding.flatten() if len(embedding.shape) > 1 else embedding
                
                import plotly.graph_objects as go
                fig = go.Figure(data=[
                    go.Histogram(
                        x=embedding_flat,
                        nbinsx=30,
                        marker_color='rgb(55, 83, 109)'
                    )
                ])
                
                fig.update_layout(
                    title="Embedding Value Distribution",
                    xaxis_title="Activation Value",
                    yaxis_title="Count",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_3d_structure(self, sequence: str):
        """Render 3D structure visualization"""
        st.subheader("🧊 3D Structure Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = self.visualizer.create_3d_structure_placeholder(sequence)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.info(
                "**Note:** This is a placeholder 3D visualization. "
                "For accurate protein structure prediction, consider using:\n"
                "- AlphaFold Database\n"
                "- ChimeraX\n"
                "- PyMOL\n"
                "- ColabFold"
            )
            
            # Download options
            if st.button("Generate PDB File"):
                st.info("PDB file generation would be implemented here")
            
            if st.button("Predict with AlphaFold"):
                st.info("AlphaFold prediction integration would be implemented here")
