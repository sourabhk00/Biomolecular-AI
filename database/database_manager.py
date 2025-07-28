import streamlit as st
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timedelta
from database.schema import *

class BiomolecularDatabaseManager:
    """Comprehensive database manager for biomolecular platform"""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    # Protein Management
    def add_protein(self, name: str, sequence: str, organism: str = None, 
                   uniprot_id: str = None, description: str = None) -> int:
        """Add new protein to database"""
        with self.get_session() as db:
            protein = Protein(
                name=name,
                sequence=sequence,
                organism=organism,
                uniprot_id=uniprot_id,
                description=description,
                molecular_weight=len(sequence) * 110,  # Approximate
                length=len(sequence)
            )
            db.add(protein)
            db.commit()
            db.refresh(protein)
            return protein.id
    
    def get_proteins(self, limit: int = 100, search: str = None) -> List[Dict]:
        """Get proteins with optional search"""
        with self.get_session() as db:
            query = db.query(Protein)
            
            if search:
                query = query.filter(
                    or_(
                        Protein.name.ilike(f"%{search}%"),
                        Protein.organism.ilike(f"%{search}%"),
                        Protein.uniprot_id.ilike(f"%{search}%")
                    )
                )
            
            proteins = query.limit(limit).all()
            
            return [
                {
                    'id': p.id,
                    'name': p.name,
                    'sequence': p.sequence[:50] + '...' if len(p.sequence) > 50 else p.sequence,
                    'organism': p.organism,
                    'uniprot_id': p.uniprot_id,
                    'length': p.length,
                    'molecular_weight': p.molecular_weight,
                    'created_at': p.created_at
                }
                for p in proteins
            ]
    
    def save_protein_analysis(self, protein_id: int, analysis_type: str, 
                            model_version: str, results: Dict, 
                            confidence_scores: Dict = None, 
                            processing_time: float = None) -> int:
        """Save protein analysis results"""
        with self.get_session() as db:
            analysis = ProteinAnalysis(
                protein_id=protein_id,
                analysis_type=analysis_type,
                model_version=model_version,
                results=results,
                confidence_scores=confidence_scores or {},
                processing_time=processing_time
            )
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            return analysis.id
    
    # Compound Management
    def add_compound(self, name: str, smiles: str = None, 
                    molecular_formula: str = None, 
                    molecular_weight: float = None,
                    compound_type: str = "drug",
                    properties: Dict = None) -> int:
        """Add new compound to database"""
        with self.get_session() as db:
            compound = Compound(
                name=name,
                smiles=smiles,
                molecular_formula=molecular_formula,
                molecular_weight=molecular_weight,
                compound_type=compound_type,
                properties=properties or {}
            )
            db.add(compound)
            db.commit()
            db.refresh(compound)
            return compound.id
    
    def add_interaction(self, protein_id: int, compound_id: int,
                       interaction_type: str, binding_affinity: float = None,
                       affinity_unit: str = "nM", predicted: bool = False,
                       prediction_confidence: float = None) -> int:
        """Add protein-compound interaction"""
        with self.get_session() as db:
            interaction = ProteinCompoundInteraction(
                protein_id=protein_id,
                compound_id=compound_id,
                interaction_type=interaction_type,
                binding_affinity=binding_affinity,
                affinity_unit=affinity_unit,
                predicted=predicted,
                prediction_confidence=prediction_confidence
            )
            db.add(interaction)
            db.commit()
            db.refresh(interaction)
            return interaction.id
    
    # Experiment Management
    def create_experiment(self, name: str, experiment_type: str,
                         target_protein_id: int = None, description: str = None,
                         parameters: Dict = None, created_by: str = None,
                         institution: str = None) -> int:
        """Create new experiment"""
        with self.get_session() as db:
            experiment = Experiment(
                name=name,
                experiment_type=experiment_type,
                target_protein_id=target_protein_id,
                description=description,
                parameters=parameters or {},
                created_by=created_by,
                institution=institution,
                start_time=datetime.utcnow()
            )
            db.add(experiment)
            db.commit()
            db.refresh(experiment)
            return experiment.id
    
    def update_experiment_status(self, experiment_id: int, status: str,
                               results: Dict = None, metrics: Dict = None):
        """Update experiment status and results"""
        with self.get_session() as db:
            experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
            if experiment:
                experiment.status = status
                if results:
                    experiment.results = results
                if metrics:
                    experiment.metrics = metrics
                if status == "completed":
                    experiment.end_time = datetime.utcnow()
                db.commit()
    
    def save_model_run(self, experiment_id: int, model_type: str,
                      model_version: str, hyperparameters: Dict,
                      performance_metrics: Dict, training_time: float = None,
                      training_data_size: int = None) -> int:
        """Save model training run"""
        with self.get_session() as db:
            model_run = ModelRun(
                experiment_id=experiment_id,
                model_type=model_type,
                model_version=model_version,
                hyperparameters=hyperparameters,
                training_data_size=training_data_size,
                performance_metrics=performance_metrics,
                training_time=training_time,
                status="completed"
            )
            db.add(model_run)
            db.commit()
            db.refresh(model_run)
            return model_run.id
    
    # Federated Learning Management
    def create_federated_session(self, session_name: str, coordinator_institution: str,
                                model_type: str, global_model_config: Dict,
                                privacy_level: str = "basic") -> int:
        """Create federated learning session"""
        with self.get_session() as db:
            session = FederatedSession(
                session_name=session_name,
                coordinator_institution=coordinator_institution,
                model_type=model_type,
                global_model_config=global_model_config,
                privacy_level=privacy_level
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.id
    
    def add_federated_participant(self, session_id: int, institution_name: str,
                                 institution_id: str, data_size: int,
                                 data_type: str = "protein_sequences") -> int:
        """Add participant to federated session"""
        with self.get_session() as db:
            participant = FederatedParticipant(
                session_id=session_id,
                institution_name=institution_name,
                institution_id=institution_id,
                data_size=data_size,
                data_type=data_type
            )
            db.add(participant)
            db.commit()
            db.refresh(participant)
            return participant.id
    
    def create_federated_round(self, session_id: int, round_number: int,
                              participants_count: int) -> int:
        """Create new federated round"""
        with self.get_session() as db:
            round_obj = FederatedRound(
                session_id=session_id,
                round_number=round_number,
                participants_count=participants_count,
                start_time=datetime.utcnow()
            )
            db.add(round_obj)
            db.commit()
            db.refresh(round_obj)
            return round_obj.id
    
    def save_participant_update(self, round_id: int, participant_id: int,
                               local_metrics: Dict, data_samples_used: int,
                               training_time: float = None) -> int:
        """Save participant model update"""
        with self.get_session() as db:
            update = ParticipantUpdate(
                round_id=round_id,
                participant_id=participant_id,
                local_metrics=local_metrics,
                data_samples_used=data_samples_used,
                training_time=training_time
            )
            db.add(update)
            db.commit()
            db.refresh(update)
            return update.id
    
    # AutoML Management
    def create_automl_experiment(self, experiment_name: str, target_task: str,
                                search_space: Dict, optimization_objective: str) -> int:
        """Create AutoML experiment"""
        with self.get_session() as db:
            automl_exp = AutoMLExperiment(
                experiment_name=experiment_name,
                target_task=target_task,
                search_space=search_space,
                optimization_objective=optimization_objective
            )
            db.add(automl_exp)
            db.commit()
            db.refresh(automl_exp)
            return automl_exp.id
    
    def save_automl_trial(self, experiment_id: int, trial_number: int,
                         architecture_config: Dict, performance_score: float,
                         training_time: float = None) -> int:
        """Save AutoML trial results"""
        with self.get_session() as db:
            trial = AutoMLTrial(
                experiment_id=experiment_id,
                trial_number=trial_number,
                architecture_config=architecture_config,
                performance_score=performance_score,
                training_time=training_time
            )
            db.add(trial)
            db.commit()
            db.refresh(trial)
            return trial.id
    
    # AI Hypothesis Management
    def save_ai_hypothesis(self, title: str, hypothesis_statement: str,
                          research_context: str, hypothesis_type: str,
                          confidence_score: float, generated_by_model: str,
                          suggested_experiments: List = None) -> int:
        """Save AI-generated hypothesis"""
        with self.get_session() as db:
            hypothesis = AIHypothesis(
                title=title,
                hypothesis_statement=hypothesis_statement,
                research_context=research_context,
                hypothesis_type=hypothesis_type,
                confidence_score=confidence_score,
                generated_by_model=generated_by_model,
                suggested_experiments=suggested_experiments or []
            )
            db.add(hypothesis)
            db.commit()
            db.refresh(hypothesis)
            return hypothesis.id
    
    # Analytics and Reporting
    def get_experiment_stats(self) -> Dict:
        """Get experiment statistics"""
        with self.get_session() as db:
            total_experiments = db.query(Experiment).count()
            completed_experiments = db.query(Experiment).filter(
                Experiment.status == "completed"
            ).count()
            
            recent_experiments = db.query(Experiment).filter(
                Experiment.created_at >= datetime.utcnow() - timedelta(days=30)
            ).count()
            
            protein_count = db.query(Protein).count()
            compound_count = db.query(Compound).count()
            interaction_count = db.query(ProteinCompoundInteraction).count()
            
            return {
                'total_experiments': total_experiments,
                'completed_experiments': completed_experiments,
                'recent_experiments': recent_experiments,
                'proteins': protein_count,
                'compounds': compound_count,
                'interactions': interaction_count
            }
    
    def get_federated_stats(self) -> Dict:
        """Get federated learning statistics"""
        with self.get_session() as db:
            active_sessions = db.query(FederatedSession).filter(
                FederatedSession.status == "active"
            ).count()
            
            total_participants = db.query(FederatedParticipant).count()
            total_rounds = db.query(FederatedRound).count()
            
            # Recent activity
            recent_rounds = db.query(FederatedRound).filter(
                FederatedRound.start_time >= datetime.utcnow() - timedelta(days=7)
            ).count()
            
            return {
                'active_sessions': active_sessions,
                'total_participants': total_participants,
                'total_rounds': total_rounds,
                'recent_rounds': recent_rounds
            }
    
    def get_model_performance_history(self, model_type: str = None) -> List[Dict]:
        """Get model performance trends"""
        with self.get_session() as db:
            query = db.query(ModelRun).join(Experiment)
            
            if model_type:
                query = query.filter(ModelRun.model_type == model_type)
            
            runs = query.order_by(desc(ModelRun.created_at)).limit(50).all()
            
            return [
                {
                    'id': run.id,
                    'model_type': run.model_type,
                    'experiment_name': run.experiment.name,
                    'performance_metrics': run.performance_metrics,
                    'training_time': run.training_time,
                    'created_at': run.created_at
                }
                for run in runs
            ]
    
    def search_database(self, query: str, table: str = None) -> List[Dict]:
        """Generic database search"""
        results = []
        
        with self.get_session() as db:
            if not table or table == "proteins":
                proteins = db.query(Protein).filter(
                    or_(
                        Protein.name.ilike(f"%{query}%"),
                        Protein.description.ilike(f"%{query}%"),
                        Protein.organism.ilike(f"%{query}%")
                    )
                ).limit(20).all()
                
                results.extend([
                    {
                        'type': 'protein',
                        'id': p.id,
                        'name': p.name,
                        'description': p.description,
                        'organism': p.organism
                    }
                    for p in proteins
                ])
            
            if not table or table == "experiments":
                experiments = db.query(Experiment).filter(
                    or_(
                        Experiment.name.ilike(f"%{query}%"),
                        Experiment.description.ilike(f"%{query}%")
                    )
                ).limit(20).all()
                
                results.extend([
                    {
                        'type': 'experiment',
                        'id': e.id,
                        'name': e.name,
                        'description': e.description,
                        'status': e.status
                    }
                    for e in experiments
                ])
        
        return results

# Global database manager instance
db_manager = BiomolecularDatabaseManager()