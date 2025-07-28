# Biomolecular Foundation Model

## Overview

This repository contains a Streamlit-based web application for biomolecular analysis and AI-powered drug design. The system integrates multiple machine learning models including ESM-2 protein analysis, hybrid transformer-GNN architectures, and molecular processing capabilities. The application provides an interactive interface for protein sequence analysis, drug design workflows, and model training pipelines.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page layout
- **UI Components**: Modular component-based architecture with separate classes for different functionalities
- **Layout**: Wide layout with sidebar navigation and tabbed interfaces
- **State Management**: Streamlit session state for maintaining application state across interactions

### Backend Architecture
- **ML Framework**: PyTorch-based neural networks with GPU acceleration support
- **Model Architecture**: Hybrid approach combining Transformer attention mechanisms with graph neural network components
- **Processing Pipeline**: Modular utility classes for molecular data processing and validation

### Component Structure
The application follows a modular component-based architecture:
- `ProteinAnalyzerComponent`: Handles protein sequence input and analysis
- `DrugDesignerComponent`: Manages AI-powered drug design workflows  
- `TrainingPipelineComponent`: Provides model training and monitoring capabilities

## Key Components

### Machine Learning Models
1. **ESMProteinAnalyzer** (`models/esm_model.py`)
   - Uses ESM-2 transformer model for protein embeddings
   - Fallback mechanism for model loading failures
   - Supports sequences up to 512 tokens with GPU acceleration

2. **HybridAffinityPredictor** (`models/hybrid_predictor.py`)
   - Transformer-GNN hybrid architecture for molecular property prediction
   - Multi-head attention mechanism with 8 attention heads
   - Property-specific prediction heads for binding affinity, solubility, toxicity, stability, and bioavailability
   - Configurable hidden dimensions (default 256) with dropout regularization

### Utility Classes
1. **MolecularProcessor** (`utils/molecular_utils.py`)
   - Validates protein and DNA sequences with comprehensive error checking
   - Supports FASTA format parsing
   - Enforces sequence length constraints (10-2000 amino acids for proteins)

2. **MolecularVisualizer** (`utils/visualization.py`)
   - Plotly-based visualization system
   - Color-coded amino acid and nucleotide representations
   - Interactive charts for sequence composition analysis

### User Interface Components
1. **Protein Analysis Interface**
   - Multiple input methods: manual entry, file upload, example sequences
   - Real-time sequence validation with detailed error messages
   - Interactive analysis results display

2. **Drug Design Interface**
   - Target protein parameter configuration
   - Binding site specification capabilities
   - Design workflow management

3. **Training Pipeline Interface**
   - Configurable training parameters
   - Real-time training monitoring
   - Training history visualization

## Data Flow

1. **Input Processing**: Users input protein sequences through multiple interfaces (manual, file upload, examples)
2. **Validation**: Sequences are validated using `MolecularProcessor` with comprehensive error checking
3. **Embedding Generation**: Valid sequences are processed through ESM-2 model to generate protein embeddings
4. **Analysis/Prediction**: Embeddings are fed into hybrid predictor for property prediction or drug design
5. **Visualization**: Results are displayed using interactive Plotly visualizations
6. **State Management**: All results and configurations are maintained in Streamlit session state

## External Dependencies

### Core ML Libraries
- **PyTorch**: Primary ML framework for model implementation
- **Transformers**: Hugging Face library for ESM-2 model integration
- **NumPy/Pandas**: Data manipulation and numerical computing

### Visualization
- **Plotly**: Interactive plotting library for molecular visualizations
- **Streamlit**: Web application framework

### Molecular Processing
- **Potential integrations**: RDKit, BioPython, DeepChem (referenced in attached assets)

### Model Hosting
- **Hugging Face Hub**: For model storage and deployment (token-based authentication)
- **Weights & Biases**: Experiment tracking (referenced in attached assets)

## Deployment Strategy

### Local Development
- Streamlit development server with hot reloading
- Environment variable configuration for API tokens
- GPU acceleration support with automatic CPU fallback

### Production Considerations
- **Model Loading**: Graceful fallback mechanisms for model loading failures
- **Resource Management**: Automatic device detection (CUDA/CPU) with memory optimization
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Scalability**: Modular architecture supports horizontal scaling of components

### Integration Capabilities
The codebase shows preparation for:
- Federated learning deployment across multiple institutions
- Hugging Face Space deployment for public access
- Integration with cloud-based model serving platforms
- Multi-modal inference and training pipelines

### Security
- Token-based authentication for external API access
- Environment variable management for sensitive credentials
- Input validation and sanitization for all user inputs