import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import os
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.esm_model import ESMProteinAnalyzer
from models.hybrid_predictor import HybridAffinityPredictor
from utils.molecular_utils import MolecularProcessor
from utils.visualization import MolecularVisualizer
from components.protein_analyzer import ProteinAnalyzerComponent
from components.drug_designer import DrugDesignerComponent
from components.training_pipeline import TrainingPipelineComponent
from components.ai_lab import AILabComponent

# Page configuration
st.set_page_config(
    page_title="Biomolecular Foundation Model",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BiomolecularApp:
    def __init__(self):
        self.initialize_session_state()
        self.setup_models()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'training_status' not in st.session_state:
            st.session_state.training_status = "idle"
    
    def setup_models(self):
        """Initialize ML models"""
        if not st.session_state.models_loaded:
            with st.spinner("Loading biomolecular models..."):
                try:
                    self.esm_analyzer = ESMProteinAnalyzer()
                    self.hybrid_predictor = HybridAffinityPredictor()
                    self.molecular_processor = MolecularProcessor()
                    self.visualizer = MolecularVisualizer()
                    st.session_state.models_loaded = True
                    st.success("Models loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading models: {str(e)}")
                    st.stop()
        else:
            self.esm_analyzer = ESMProteinAnalyzer()
            self.hybrid_predictor = HybridAffinityPredictor()
            self.molecular_processor = MolecularProcessor()
            self.visualizer = MolecularVisualizer()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.title("🧬 Biomolecular AI")
            st.markdown("---")
            
            mode = st.selectbox(
                "Select Analysis Mode",
                ["Protein Analysis", "Drug Design", "AI Laboratory", "Federated Learning", "Database", "Model Training", "Batch Processing"],
                key="analysis_mode"
            )
            
            st.markdown("---")
            st.subheader("Model Information")
            st.info("ESM-2 Protein Language Model")
            st.info("Hybrid Transformer-GNN Architecture")
            st.info("Real-time Molecular Property Prediction")
            
            st.markdown("---")
            st.subheader("Quick Stats")
            if st.session_state.analysis_results:
                st.metric("Analyses Completed", len(st.session_state.analysis_results))
            else:
                st.metric("Analyses Completed", 0)
            
            return mode
    
    def render_header(self):
        """Render main header"""
        st.title("🧬 Biomolecular Foundation Model")
        st.markdown("""
        Advanced AI platform for protein analysis, molecular property prediction, and drug design.
        Powered by ESM-2 language models and hybrid neural architectures.
        """)
        st.markdown("---")
    
    def run(self):
        """Main application loop"""
        self.render_header()
        mode = self.render_sidebar()
        
        # Main content area
        if mode == "Protein Analysis":
            self.render_protein_analysis()
        elif mode == "Drug Design":
            self.render_drug_design()
        elif mode == "AI Laboratory":
            self.render_ai_laboratory()
        elif mode == "Federated Learning":
            self.render_federated_learning()
        elif mode == "Database":
            self.render_database()
        elif mode == "Model Training":
            self.render_model_training()
        elif mode == "Batch Processing":
            self.render_batch_processing()
    
    def render_protein_analysis(self):
        """Render protein analysis interface"""
        st.header("🔬 Protein Sequence Analysis")
        
        analyzer_component = ProteinAnalyzerComponent(
            self.esm_analyzer, 
            self.visualizer
        )
        analyzer_component.render()
    
    def render_drug_design(self):
        """Render drug design interface"""
        st.header("💊 AI-Powered Drug Design")
        
        designer_component = DrugDesignerComponent(
            self.hybrid_predictor,
            self.molecular_processor,
            self.visualizer
        )
        designer_component.render()
    
    def render_ai_laboratory(self):
        """Render AI laboratory interface"""
        ai_lab = AILabComponent()
        ai_lab.render()
    
    def render_federated_learning(self):
        """Render federated learning interface"""
        from components.federated_lab import FederatedLabComponent
        federated_lab = FederatedLabComponent()
        federated_lab.render()
    
    def render_database(self):
        """Render database interface"""
        from components.database_interface import DatabaseInterfaceComponent
        database_interface = DatabaseInterfaceComponent()
        database_interface.render()
    
    def render_model_training(self):
        """Render model training interface"""
        st.header("🎯 Model Training Pipeline")
        
        training_component = TrainingPipelineComponent(
            self.hybrid_predictor
        )
        training_component.render()
    
    def render_batch_processing(self):
        """Render batch processing interface"""
        st.header("📋 Batch Processing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Data")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['csv', 'fasta', 'pdb', 'sdf'],
                help="Upload protein sequences, molecular structures, or datasets"
            )
            
            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1].lower()
                
                if file_type == 'csv':
                    df = pd.read_csv(uploaded_file)
                    st.write("Data preview:")
                    st.dataframe(df.head())
                    
                    if st.button("Process Batch"):
                        self.process_batch_data(df)
                
                elif file_type == 'fasta':
                    content = str(uploaded_file.read(), "utf-8")
                    sequences = self.molecular_processor.parse_fasta(content)
                    st.write(f"Found {len(sequences)} sequences")
                    
                    if st.button("Analyze Sequences"):
                        self.process_fasta_batch(sequences)
        
        with col2:
            st.subheader("Results")
            if 'batch_results' in st.session_state:
                results_df = pd.DataFrame(st.session_state.batch_results)
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="batch_analysis_results.csv",
                    mime="text/csv"
                )
    
    def process_batch_data(self, df):
        """Process batch data"""
        with st.spinner("Processing batch data..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, row in df.iterrows():
                # Process each row based on data type
                if 'sequence' in df.columns:
                    embedding = self.esm_analyzer.get_embedding(row['sequence'])
                    prediction = self.hybrid_predictor.predict(embedding)
                    
                    results.append({
                        'index': i,
                        'sequence': row['sequence'][:20] + "...",
                        'prediction': prediction,
                        'confidence': np.random.uniform(0.8, 0.99)  # Placeholder
                    })
                
                progress_bar.progress((i + 1) / len(df))
            
            st.session_state.batch_results = results
            st.success(f"Processed {len(results)} items successfully!")
    
    def process_fasta_batch(self, sequences):
        """Process FASTA sequences"""
        with st.spinner("Analyzing sequences..."):
            results = []
            progress_bar = st.progress(0)
            
            for i, (name, seq) in enumerate(sequences.items()):
                embedding = self.esm_analyzer.get_embedding(seq)
                prediction = self.hybrid_predictor.predict(embedding)
                
                results.append({
                    'name': name,
                    'length': len(seq),
                    'prediction': prediction,
                    'stability_score': np.random.uniform(0.6, 0.95)  # Placeholder
                })
                
                progress_bar.progress((i + 1) / len(sequences))
            
            st.session_state.batch_results = results
            st.success(f"Analyzed {len(results)} sequences successfully!")

def main():
    """Main function"""
    try:
        app = BiomolecularApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
