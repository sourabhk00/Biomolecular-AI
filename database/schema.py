import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json


# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Protein(Base):
    """Protein sequence and metadata storage"""
    __tablename__ = "proteins"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True)
    sequence = Column(Text, nullable=False)
    organism = Column(String(200))
    uniprot_id = Column(String(50), unique=True, index=True)
    description = Column(Text)
    molecular_weight = Column(Float)
    length = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analyses = relationship("ProteinAnalysis", back_populates="protein")
    experiments = relationship("Experiment", back_populates="target_protein")

class ProteinAnalysis(Base):
    """Results from protein analysis"""
    __tablename__ = "protein_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    protein_id = Column(Integer, ForeignKey("proteins.id"))
    analysis_type = Column(String(100))  # ESM, LLM, Structure, etc.
    model_version = Column(String(100))
    results = Column(JSON)  # Store analysis results as JSON
    confidence_scores = Column(JSON)
    processing_time = Column(Float)  # seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    protein = relationship("Protein", back_populates="analyses")

class Compound(Base):
    """Drug compounds and molecules"""
    __tablename__ = "compounds"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), index=True)
    smiles = Column(Text)  # SMILES notation
    inchi = Column(Text)   # InChI notation
    molecular_formula = Column(String(200))
    molecular_weight = Column(Float)
    compound_type = Column(String(100))  # drug, metabolite, natural_product, etc.
    source = Column(String(200))  # ChEMBL, PubChem, etc.
    external_id = Column(String(100))
    properties = Column(JSON)  # ADMET and other properties
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    interactions = relationship("ProteinCompoundInteraction", back_populates="compound")

class ProteinCompoundInteraction(Base):
    """Protein-compound binding interactions"""
    __tablename__ = "protein_compound_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    protein_id = Column(Integer, ForeignKey("proteins.id"))
    compound_id = Column(Integer, ForeignKey("compounds.id"))
    interaction_type = Column(String(100))  # binding, inhibition, activation
    binding_affinity = Column(Float)  # IC50, Kd, Ki values
    affinity_unit = Column(String(20))  # nM, uM, mM
    activity_value = Column(Float)
    activity_unit = Column(String(20))
    assay_type = Column(String(200))
    publication_doi = Column(String(200))
    experimental_conditions = Column(JSON)
    predicted = Column(Boolean, default=False)  # True if AI-predicted
    prediction_confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    protein = relationship("Protein")
    compound = relationship("Compound", back_populates="interactions")

class Experiment(Base):
    """Experimental records and workflows"""
    __tablename__ = "experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    experiment_type = Column(String(100))  # drug_discovery, protein_analysis, federated_learning
    target_protein_id = Column(Integer, ForeignKey("proteins.id"))
    description = Column(Text)
    status = Column(String(50), default="created")  # created, running, completed, failed
    parameters = Column(JSON)  # Experiment configuration
    results = Column(JSON)     # Experiment results
    metrics = Column(JSON)     # Performance metrics
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    created_by = Column(String(200))
    institution = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    target_protein = relationship("Protein", back_populates="experiments")
    model_runs = relationship("ModelRun", back_populates="experiment")

class ModelRun(Base):
    """Individual model training/inference runs"""
    __tablename__ = "model_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"))
    model_type = Column(String(100))  # BiomolecularLLM, ESM, HybridPredictor, etc.
    model_version = Column(String(100))
    hyperparameters = Column(JSON)
    training_data_size = Column(Integer)
    validation_data_size = Column(Integer)
    performance_metrics = Column(JSON)  # accuracy, loss, etc.
    model_artifacts = Column(JSON)  # paths to saved models, weights
    training_time = Column(Float)  # seconds
    inference_time = Column(Float)  # seconds
    gpu_memory_used = Column(Float)  # MB
    status = Column(String(50), default="initialized")
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="model_runs")

class FederatedSession(Base):
    """Federated learning sessions"""
    __tablename__ = "federated_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String(200), nullable=False)
    coordinator_institution = Column(String(200))
    model_type = Column(String(100))
    global_model_config = Column(JSON)
    privacy_level = Column(String(50))  # basic, differential_privacy, secure_aggregation
    total_rounds = Column(Integer, default=0)
    active_participants = Column(Integer, default=0)
    status = Column(String(50), default="created")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    rounds = relationship("FederatedRound", back_populates="session")
    participants = relationship("FederatedParticipant", back_populates="session")

class FederatedRound(Base):
    """Individual federated learning rounds"""
    __tablename__ = "federated_rounds"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("federated_sessions.id"))
    round_number = Column(Integer, nullable=False)
    participants_count = Column(Integer)
    aggregation_method = Column(String(100))
    global_metrics = Column(JSON)
    convergence_score = Column(Float)
    privacy_budget_spent = Column(Float)
    round_duration = Column(Float)  # seconds
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    status = Column(String(50), default="started")
    
    # Relationships
    session = relationship("FederatedSession", back_populates="rounds")
    participant_updates = relationship("ParticipantUpdate", back_populates="round")

class FederatedParticipant(Base):
    """Institutions participating in federated learning"""
    __tablename__ = "federated_participants"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("federated_sessions.id"))
    institution_name = Column(String(200), nullable=False)
    institution_id = Column(String(100), unique=True)
    data_size = Column(Integer)
    data_type = Column(String(100))
    privacy_preferences = Column(JSON)
    compute_resources = Column(JSON)
    join_date = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime)
    status = Column(String(50), default="active")
    
    # Relationships
    session = relationship("FederatedSession", back_populates="participants")
    updates = relationship("ParticipantUpdate", back_populates="participant")

class ParticipantUpdate(Base):
    """Model updates from federated participants"""
    __tablename__ = "participant_updates"
    
    id = Column(Integer, primary_key=True, index=True)
    round_id = Column(Integer, ForeignKey("federated_rounds.id"))
    participant_id = Column(Integer, ForeignKey("federated_participants.id"))
    local_metrics = Column(JSON)
    data_samples_used = Column(Integer)
    training_time = Column(Float)
    communication_cost = Column(Float)  # bytes transferred
    privacy_noise_scale = Column(Float)
    model_update_hash = Column(String(256))  # for integrity verification
    submitted_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    round = relationship("FederatedRound", back_populates="participant_updates")
    participant = relationship("FederatedParticipant", back_populates="updates")

class LiteratureReference(Base):
    """Scientific literature and citations"""
    __tablename__ = "literature_references"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    authors = Column(Text)
    journal = Column(String(200))
    publication_year = Column(Integer)
    doi = Column(String(200), unique=True, index=True)
    pubmed_id = Column(String(50), unique=True, index=True)
    abstract = Column(Text)
    keywords = Column(JSON)
    research_areas = Column(JSON)
    citation_count = Column(Integer, default=0)
    relevance_score = Column(Float)  # AI-computed relevance
    created_at = Column(DateTime, default=datetime.utcnow)

class AIHypothesis(Base):
    """AI-generated research hypotheses"""
    __tablename__ = "ai_hypotheses"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False)
    hypothesis_statement = Column(Text, nullable=False)
    research_context = Column(Text)
    hypothesis_type = Column(String(100))  # mechanistic, predictive, causal, etc.
    evidence_level = Column(String(50))
    confidence_score = Column(Float)
    testability_score = Column(Float)
    novelty_score = Column(Float)
    generated_by_model = Column(String(100))  # GPT-4, Claude, etc.
    supporting_literature = Column(JSON)  # Literature reference IDs
    suggested_experiments = Column(JSON)
    validation_status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)

class AutoMLExperiment(Base):
    """AutoML architecture search experiments"""
    __tablename__ = "automl_experiments"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(200), nullable=False)
    target_task = Column(String(200))  # protein_prediction, drug_discovery, etc.
    search_space = Column(JSON)
    optimization_objective = Column(String(100))  # accuracy, speed, efficiency
    total_trials = Column(Integer)
    completed_trials = Column(Integer, default=0)
    best_architecture = Column(JSON)
    best_score = Column(Float)
    search_time = Column(Float)  # seconds
    status = Column(String(50), default="created")
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    trials = relationship("AutoMLTrial", back_populates="experiment")

class AutoMLTrial(Base):
    """Individual AutoML trials"""
    __tablename__ = "automl_trials"
    
    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("automl_experiments.id"))
    trial_number = Column(Integer)
    architecture_config = Column(JSON)
    hyperparameters = Column(JSON)
    performance_score = Column(Float)
    training_time = Column(Float)
    model_size = Column(Float)  # MB
    inference_time = Column(Float)  # ms
    memory_usage = Column(Float)  # MB
    trial_status = Column(String(50), default="completed")
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment = relationship("AutoMLExperiment", back_populates="trials")

# Create all tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")
