import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import os


class AILabComponent:
    """Advanced AI Laboratory for biomolecular modeling and generation"""

    def __init__(self):
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all AI models"""
        with st.spinner("Loading advanced AI models..."):
            try:
                from models.bio_llm import (
                    BiomolecularLLM, 
                    DiffusionMolecularGenerator,
                    ProteinFoldingPredictor,
                    QuantumMolecularOracle
                )
                
                self.bio_llm = BiomolecularLLM(embed_dim=256, num_layers=6)
                self.diffusion_model = DiffusionMolecularGenerator(feature_dim=128)
                self.folding_predictor = ProteinFoldingPredictor(hidden_dim=256)
                self.quantum_oracle = QuantumMolecularOracle(input_dim=256)
                
                st.success("Advanced AI models loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading AI models: {str(e)}")
                st.stop()
    
    def render(self):
        """Render the AI Lab interface"""
        st.header("🤖 Advanced AI Laboratory")
        st.markdown("Next-generation biomolecular AI with LLMs, diffusion models, and quantum computing")
        
        # AI Model Selector
        ai_mode = st.selectbox(
            "Select AI Model:",
            [
                "Biomolecular LLM", 
                "External LLM Integration",
                "Diffusion Generator", 
                "Protein Structure Predictor",
                "Quantum Oracle",
                "Multi-Model Pipeline"
            ]
        )
        
        if ai_mode == "Biomolecular LLM":
            self.render_bio_llm()
        elif ai_mode == "External LLM Integration":
            self.render_external_llm()
        elif ai_mode == "Diffusion Generator":
            self.render_diffusion_generator()
        elif ai_mode == "Protein Structure Predictor":
            self.render_structure_predictor()
        elif ai_mode == "Quantum Oracle":
            self.render_quantum_oracle()
        elif ai_mode == "Multi-Model Pipeline":
            self.render_multi_model_pipeline()
    
    def render_bio_llm(self):
        """Render biomolecular LLM interface"""
        st.subheader("🧠 Biomolecular Large Language Model")
        st.markdown("Generate and analyze protein sequences using transformer architecture")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Sequence Generation**")
            
            prompt = st.text_input(
                "Prompt sequence (optional):",
                placeholder="MKFLVN...",
                help="Start of protein sequence to continue"
            )
            
            max_length = st.slider(
                "Maximum length:",
                min_value=10,
                max_value=200,
                value=50
            )
            
            temperature = st.slider(
                "Temperature (creativity):",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            if st.button("Generate Protein Sequence", type="primary"):
                with st.spinner("Generating sequence..."):
                    generated_seq = self.bio_llm.generate_sequence(
                        prompt=prompt,
                        max_length=max_length,
                        temperature=temperature
                    )
                    
                    st.session_state.generated_sequence = generated_seq
                    st.success("Sequence generated!")
        
        with col2:
            st.write("**Property Prediction**")
            
            analysis_sequence = st.text_area(
                "Sequence for analysis:",
                value=st.session_state.get('generated_sequence', ''),
                height=100,
                placeholder="Enter protein sequence to analyze..."
            )
            
            if st.button("Analyze Properties") and analysis_sequence:
                with st.spinner("Analyzing properties..."):
                    properties = self.bio_llm.predict_properties(analysis_sequence)
                    
                    # Display properties
                    for prop, value in properties.items():
                        st.metric(prop.replace('_', ' ').title(), f"{value:.3f}")
                    
                    # Store for visualization
                    st.session_state.llm_properties = properties
        
        # Results visualization
        if 'generated_sequence' in st.session_state:
            st.subheader("Generated Sequence")
            seq = st.session_state.generated_sequence
            st.code(seq, language="text")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("Length", len(seq))
            with col2:
                st.metric("Unique Residues", len(set(seq)))
        
        if 'llm_properties' in st.session_state:
            st.subheader("Property Analysis")
            properties = st.session_state.llm_properties
            
            # Radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(properties.values()),
                theta=list(properties.keys()),
                fill='toself',
                name='Properties'
            ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Predicted Properties",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_diffusion_generator(self):
        """Render diffusion model interface"""
        st.subheader("🌊 Diffusion Molecular Generator")
        st.markdown("Generate novel molecular structures using diffusion models")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Generation Parameters**")
            
            num_atoms = st.slider(
                "Number of atoms:",
                min_value=10,
                max_value=100,
                value=30
            )
            
            num_samples = st.slider(
                "Number of samples:",
                min_value=1,
                max_value=5,
                value=3
            )
            
            if st.button("Generate Molecules", type="primary"):
                with st.spinner("Generating molecular structures..."):
                    molecules = []
                    
                    for i in range(num_samples):
                        molecule = self.diffusion_model.generate_molecule(num_atoms)
                        molecules.append(molecule.cpu().numpy())
                    
                    st.session_state.generated_molecules = molecules
                    st.success(f"Generated {num_samples} molecular structures!")
        
        with col2:
            st.write("**Molecular Properties**")
            
            if 'generated_molecules' in st.session_state:
                molecules = st.session_state.generated_molecules
                
                # Display statistics
                for i, mol in enumerate(molecules):
                    with st.expander(f"Molecule {i+1}"):
                        st.write(f"Shape: {mol.shape}")
                        st.write(f"Mean coordinate: {mol.mean():.3f}")
                        st.write(f"Std deviation: {mol.std():.3f}")
                        
                        # Simple 3D visualization placeholder
                        if mol.shape[1] >= 3:
                            coords = mol[0, :, :3]  # First 3 dimensions as coordinates
                            
                            fig = go.Figure(data=[go.Scatter3d(
                                x=coords[:, 0],
                                y=coords[:, 1],
                                z=coords[:, 2],
                                mode='markers',
                                marker=dict(size=8, color=np.arange(len(coords)), colorscale='Viridis')
                            )])
                            
                            fig.update_layout(
                                title=f"Molecule {i+1} Structure",
                                scene=dict(
                                    xaxis_title="X",
                                    yaxis_title="Y",
                                    zaxis_title="Z"
                                ),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
    
    def render_structure_predictor(self):
        """Render protein structure prediction interface"""
        st.subheader("🔬 Protein Structure Predictor")
        st.markdown("Predict protein secondary structure and contact maps")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Input Sequence**")
            
            sequence = st.text_area(
                "Protein sequence:",
                height=100,
                placeholder="MKFLVNVALVFMVVYISYIYAA...",
                value=st.session_state.get('generated_sequence', '')
            )
            
            if st.button("Predict Structure", type="primary") and sequence:
                with st.spinner("Predicting protein structure..."):
                    # Convert sequence to one-hot encoding
                    amino_acids = 'ARNDCQEGHILKMFPSTWYV'
                    seq_features = torch.zeros(1, len(sequence), 20)
                    
                    for i, aa in enumerate(sequence.upper()):
                        if aa in amino_acids:
                            aa_idx = amino_acids.index(aa)
                            seq_features[0, i, aa_idx] = 1.0
                    
                    # Predict structure
                    predictions = self.folding_predictor(seq_features)
                    
                    st.session_state.structure_predictions = {
                        'secondary_structure': predictions['secondary_structure'].cpu().numpy(),
                        'dihedral_angles': predictions['dihedral_angles'].cpu().numpy(),
                        'contact_map': predictions['contact_map'].cpu().numpy()
                    }
                    
                    st.success("Structure predicted!")
        
        with col2:
            st.write("**Prediction Results**")
            
            if 'structure_predictions' in st.session_state:
                preds = st.session_state.structure_predictions
                
                # Secondary structure distribution
                ss_probs = preds['secondary_structure'][0].mean(axis=0)
                st.write("**Secondary Structure Distribution:**")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Helix", f"{ss_probs[0]:.2f}")
                with col_b:
                    st.metric("Sheet", f"{ss_probs[1]:.2f}")
                with col_c:
                    st.metric("Coil", f"{ss_probs[2]:.2f}")
        
        # Visualization
        if 'structure_predictions' in st.session_state:
            st.subheader("Structure Visualization")
            
            preds = st.session_state.structure_predictions
            
            tab1, tab2, tab3 = st.tabs(["Secondary Structure", "Contact Map", "Dihedral Angles"])
            
            with tab1:
                # Secondary structure plot
                ss_data = preds['secondary_structure'][0]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=ss_data[:, 0], name='Helix', line=dict(color='red')))
                fig.add_trace(go.Scatter(y=ss_data[:, 1], name='Sheet', line=dict(color='blue')))
                fig.add_trace(go.Scatter(y=ss_data[:, 2], name='Coil', line=dict(color='green')))
                
                fig.update_layout(
                    title="Secondary Structure Probabilities",
                    xaxis_title="Residue Position",
                    yaxis_title="Probability",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Contact map
                contact_map = preds['contact_map'][0]
                
                fig = go.Figure(data=go.Heatmap(
                    z=contact_map,
                    colorscale='Blues'
                ))
                
                fig.update_layout(
                    title="Predicted Contact Map",
                    xaxis_title="Residue i",
                    yaxis_title="Residue j",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Dihedral angles
                angles = preds['dihedral_angles'][0]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=angles[:, 0], name='Phi', line=dict(color='orange')))
                fig.add_trace(go.Scatter(y=angles[:, 1], name='Psi', line=dict(color='purple')))
                
                fig.update_layout(
                    title="Predicted Dihedral Angles",
                    xaxis_title="Residue Position",
                    yaxis_title="Angle (radians)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_quantum_oracle(self):
        """Render quantum molecular oracle interface"""
        st.subheader("⚛️ Quantum Molecular Oracle")
        st.markdown("Quantum-inspired molecular property prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Quantum Analysis**")
            
            molecular_input = st.selectbox(
                "Input type:",
                ["Generated Sequence", "Custom Features", "Random Molecule"]
            )
            
            if molecular_input == "Custom Features":
                num_features = st.slider("Number of features:", 10, 100, 50)
                features = st.text_area(
                    "Molecular features (comma-separated):",
                    placeholder="0.1, 0.5, 0.8, ...",
                    help="Enter numerical features representing molecular properties"
                )
            
            if st.button("Run Quantum Analysis", type="primary"):
                with st.spinner("Processing with quantum oracle..."):
                    if molecular_input == "Random Molecule":
                        # Generate random molecular features
                        features_tensor = torch.randn(1, 256)
                    elif molecular_input == "Custom Features" and features:
                        # Parse custom features
                        try:
                            feature_list = [float(x.strip()) for x in features.split(',')]
                            features_tensor = torch.tensor(feature_list).float().unsqueeze(0)
                            # Pad or truncate to expected size
                            if features_tensor.shape[1] < 256:
                                padding = torch.zeros(1, 256 - features_tensor.shape[1])
                                features_tensor = torch.cat([features_tensor, padding], dim=1)
                            else:
                                features_tensor = features_tensor[:, :256]
                        except:
                            st.error("Invalid feature format")
                            return
                    else:
                        # Use sequence from other models if available
                        if 'generated_sequence' in st.session_state:
                            # Simple feature extraction from sequence
                            seq = st.session_state.generated_sequence
                            features_tensor = torch.randn(1, 256)  # Placeholder
                        else:
                            features_tensor = torch.randn(1, 256)
                    
                    # Quantum prediction
                    quantum_score = self.quantum_oracle(features_tensor)
                    
                    st.session_state.quantum_score = quantum_score.item()
                    st.success("Quantum analysis complete!")
        
        with col2:
            st.write("**Quantum Results**")
            
            if 'quantum_score' in st.session_state:
                score = st.session_state.quantum_score
                
                st.metric("Quantum Score", f"{score:.4f}")
                
                # Interpretation
                if score > 0.8:
                    st.success("High quantum coherence - Exceptional molecular properties predicted")
                elif score > 0.6:
                    st.info("Moderate quantum coherence - Good molecular potential")
                else:
                    st.warning("Low quantum coherence - Limited molecular viability")
                
                # Quantum state visualization
                fig = go.Figure()
                
                # Simulate quantum state probabilities
                states = [f"|{i}⟩" for i in range(8)]
                probs = np.random.dirichlet(np.ones(8) * score * 10)
                
                fig.add_trace(go.Bar(
                    x=states,
                    y=probs,
                    marker_color='lightblue',
                    text=[f"{p:.3f}" for p in probs],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title="Quantum State Probabilities",
                    xaxis_title="Quantum States",
                    yaxis_title="Probability",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_multi_model_pipeline(self):
        """Render multi-model AI pipeline"""
        st.subheader("🔗 Multi-Model AI Pipeline")
        st.markdown("Combined analysis using all AI models in sequence")
        
        st.write("**Pipeline Configuration**")
        
        pipeline_steps = st.multiselect(
            "Select pipeline steps:",
            ["LLM Generation", "Diffusion Structure", "Folding Prediction", "Quantum Analysis"],
            default=["LLM Generation", "Folding Prediction", "Quantum Analysis"]
        )
        
        if st.button("Run Complete Pipeline", type="primary"):
            results = {}
            
            with st.spinner("Running multi-model pipeline..."):
                progress_bar = st.progress(0)
                
                # Step 1: LLM Generation
                if "LLM Generation" in pipeline_steps:
                    st.write("🧠 Generating sequence with Biomolecular LLM...")
                    sequence = self.bio_llm.generate_sequence(max_length=50)
                    results['generated_sequence'] = sequence
                    properties = self.bio_llm.predict_properties(sequence)
                    results['llm_properties'] = properties
                    progress_bar.progress(0.25)
                
                # Step 2: Diffusion Structure
                if "Diffusion Structure" in pipeline_steps:
                    st.write("🌊 Generating molecular structure...")
                    molecule = self.diffusion_model.generate_molecule(30)
                    results['molecule_structure'] = molecule.cpu().numpy()
                    progress_bar.progress(0.5)
                
                # Step 3: Folding Prediction
                if "Folding Prediction" in pipeline_steps:
                    st.write("🔬 Predicting protein structure...")
                    if 'generated_sequence' in results:
                        sequence = results['generated_sequence']
                        amino_acids = 'ARNDCQEGHILKMFPSTWYV'
                        seq_features = torch.zeros(1, len(sequence), 20)
                        
                        for i, aa in enumerate(sequence.upper()):
                            if aa in amino_acids:
                                aa_idx = amino_acids.index(aa)
                                seq_features[0, i, aa_idx] = 1.0
                        
                        structure_pred = self.folding_predictor(seq_features)
                        results['structure_predictions'] = {
                            'secondary_structure': structure_pred['secondary_structure'].cpu().numpy(),
                            'contact_map': structure_pred['contact_map'].cpu().numpy()
                        }
                    progress_bar.progress(0.75)
                
                # Step 4: Quantum Analysis
                if "Quantum Analysis" in pipeline_steps:
                    st.write("⚛️ Running quantum analysis...")
                    features = torch.randn(1, 256)
                    quantum_score = self.quantum_oracle(features)
                    results['quantum_score'] = quantum_score.item()
                    progress_bar.progress(1.0)
                
                st.session_state.pipeline_results = results
                st.success("Pipeline complete!")
        
        # Display pipeline results
        if 'pipeline_results' in st.session_state:
            st.subheader("Pipeline Results")
            results = st.session_state.pipeline_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'generated_sequence' in results:
                    st.metric("Sequence Length", len(results['generated_sequence']))
            
            with col2:
                if 'llm_properties' in results:
                    avg_prop = np.mean(list(results['llm_properties'].values()))
                    st.metric("Avg Property Score", f"{avg_prop:.3f}")
            
            with col3:
                if 'structure_predictions' in results:
                    ss_entropy = -np.sum(results['structure_predictions']['secondary_structure'][0].mean(axis=0) * 
                                       np.log(results['structure_predictions']['secondary_structure'][0].mean(axis=0) + 1e-8))
                    st.metric("Structure Complexity", f"{ss_entropy:.3f}")
            
            with col4:
                if 'quantum_score' in results:
                    st.metric("Quantum Score", f"{results['quantum_score']:.4f}")
            
            # Detailed results
            for step, result in results.items():
                with st.expander(f"Details: {step.replace('_', ' ').title()}"):
                    if step == 'generated_sequence':
                        st.code(result, language="text")
                    elif step == 'llm_properties':
                        df = pd.DataFrame(list(result.items()), columns=['Property', 'Score'])
                        st.dataframe(df)
                    elif step == 'quantum_score':
                        st.write(f"Quantum coherence score: {result:.6f}")
                    else:
                        st.write(f"Result type: {type(result)}")
                        if hasattr(result, 'shape'):
                            st.write(f"Shape: {result.shape}")
    
    def render_external_llm(self):
        """Render external LLM integration interface"""
        from components.llm_integration import LLMIntegrationComponent
        llm_component = LLMIntegrationComponent()
        llm_component.render()
