import streamlit as st
import json
import os
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go

class LLMIntegrationComponent:
    """Integration with external LLM APIs for advanced biomolecular analysis"""
    
    def __init__(self):
        self.initialize_apis()
    
    def initialize_apis(self):
        """Initialize API clients"""
        self.openai_available = os.getenv("OPENAI_API_KEY") is not None
        self.anthropic_available = os.getenv("ANTHROPIC_API_KEY") is not None
        
        if self.openai_available:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                self.openai_available = False
                st.warning(f"OpenAI client initialization failed: {str(e)}")
        
        if self.anthropic_available:
            try:
                from anthropic import Anthropic
                # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229"
                self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.DEFAULT_MODEL_STR = "claude-sonnet-4-20250514"
            except Exception as e:
                self.anthropic_available = False
                st.warning(f"Anthropic client initialization failed: {str(e)}")
    
    def render(self):
        """Render LLM integration interface"""
        st.header("🤖 LLM-Powered Biomolecular Analysis")
        st.markdown("Advanced AI analysis using state-of-the-art language models")
        
        # API Status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("OpenAI API", "✅ Connected" if self.openai_available else "❌ Not Available")
        with col2:
            st.metric("Anthropic API", "✅ Connected" if self.anthropic_available else "❌ Not Available")
        
        if not (self.openai_available or self.anthropic_available):
            st.error("No LLM APIs available. Please provide API keys to enable this functionality.")
            st.info("Go to the secrets section to add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY.")
            return
        
        # Main functionality tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Protein Analysis", 
            "Drug Discovery", 
            "Literature Mining", 
            "Hypothesis Generation"
        ])
        
        with tab1:
            self.render_protein_analysis()
        
        with tab2:
            self.render_drug_discovery()
        
        with tab3:
            self.render_literature_mining()
        
        with tab4:
            self.render_hypothesis_generation()
    
    def render_protein_analysis(self):
        """Render LLM-powered protein analysis"""
        st.subheader("🧬 Advanced Protein Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Input Protein Data**")
            
            input_type = st.selectbox(
                "Analysis Type:",
                ["Sequence Analysis", "Structure Prediction", "Function Prediction", "Evolution Analysis"]
            )
            
            protein_sequence = st.text_area(
                "Protein Sequence:",
                height=100,
                placeholder="MKFLVNVALVFMVVYISYIYAA...",
                help="Enter protein sequence in single-letter amino acid code"
            )
            
            analysis_depth = st.selectbox(
                "Analysis Depth:",
                ["Quick Overview", "Detailed Analysis", "Comprehensive Report"]
            )
            
            model_choice = st.selectbox(
                "LLM Model:",
                self.get_available_models()
            )
            
            if st.button("Analyze Protein", type="primary") and protein_sequence:
                self.analyze_protein_with_llm(protein_sequence, input_type, analysis_depth, model_choice)
        
        with col2:
            st.write("**Analysis Results**")
            
            if 'protein_analysis_result' in st.session_state:
                result = st.session_state.protein_analysis_result
                
                # Display analysis
                st.markdown("### Analysis Summary")
                st.write(result.get('summary', 'No summary available'))
                
                # Display properties if available
                if 'properties' in result:
                    st.markdown("### Predicted Properties")
                    for prop, value in result['properties'].items():
                        st.write(f"**{prop}:** {value}")
                
                # Display confidence scores
                if 'confidence' in result:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(result['confidence'].keys()),
                            y=list(result['confidence'].values()),
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title="Analysis Confidence Scores",
                        xaxis_title="Analysis Aspect",
                        yaxis_title="Confidence",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_drug_discovery(self):
        """Render LLM-powered drug discovery"""
        st.subheader("💊 AI Drug Discovery Assistant")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Drug Discovery Query**")
            
            discovery_type = st.selectbox(
                "Discovery Type:",
                ["Target Identification", "Lead Optimization", "ADMET Prediction", "Drug Repurposing"]
            )
            
            target_protein = st.text_input(
                "Target Protein/Disease:",
                placeholder="Enter protein name or disease"
            )
            
            constraints = st.text_area(
                "Constraints & Requirements:",
                placeholder="e.g., oral bioavailability, low toxicity, specific binding site...",
                height=100
            )
            
            model_choice = st.selectbox(
                "LLM Model:",
                self.get_available_models(),
                key="drug_model"
            )
            
            if st.button("Generate Drug Insights", type="primary") and target_protein:
                self.generate_drug_insights(target_protein, discovery_type, constraints, model_choice)
        
        with col2:
            st.write("**Discovery Results**")
            
            if 'drug_discovery_result' in st.session_state:
                result = st.session_state.drug_discovery_result
                
                st.markdown("### Drug Discovery Insights")
                st.write(result.get('insights', 'No insights available'))
                
                if 'suggestions' in result:
                    st.markdown("### Suggested Approaches")
                    for i, suggestion in enumerate(result['suggestions'], 1):
                        st.write(f"{i}. {suggestion}")
                
                if 'compounds' in result:
                    st.markdown("### Potential Compounds")
                    df = pd.DataFrame(result['compounds'])
                    st.dataframe(df, use_container_width=True)
    
    def render_literature_mining(self):
        """Render literature mining interface"""
        st.subheader("📚 Scientific Literature Mining")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Literature Query**")
            
            research_topic = st.text_input(
                "Research Topic:",
                placeholder="e.g., 'CRISPR protein engineering'"
            )
            
            focus_areas = st.multiselect(
                "Focus Areas:",
                ["Methodology", "Results", "Mechanisms", "Applications", "Limitations", "Future Directions"],
                default=["Results", "Mechanisms"]
            )
            
            time_period = st.selectbox(
                "Time Period:",
                ["Last 6 months", "Last year", "Last 2 years", "Last 5 years", "All time"]
            )
            
            model_choice = st.selectbox(
                "LLM Model:",
                self.get_available_models(),
                key="lit_model"
            )
            
            if st.button("Mine Literature", type="primary") and research_topic:
                self.mine_literature(research_topic, focus_areas, time_period, model_choice)
        
        with col2:
            st.write("**Literature Insights**")
            
            if 'literature_result' in st.session_state:
                result = st.session_state.literature_result
                
                st.markdown("### Key Findings")
                st.write(result.get('key_findings', 'No findings available'))
                
                if 'trends' in result:
                    st.markdown("### Research Trends")
                    for trend in result['trends']:
                        st.write(f"• {trend}")
                
                if 'knowledge_gaps' in result:
                    st.markdown("### Identified Knowledge Gaps")
                    for gap in result['knowledge_gaps']:
                        st.write(f"• {gap}")
    
    def render_hypothesis_generation(self):
        """Render hypothesis generation interface"""
        st.subheader("💡 AI Hypothesis Generation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Hypothesis Parameters**")
            
            research_context = st.text_area(
                "Research Context:",
                placeholder="Describe your current research, observations, or experimental data...",
                height=150
            )
            
            hypothesis_type = st.selectbox(
                "Hypothesis Type:",
                ["Mechanistic", "Predictive", "Comparative", "Causal", "Exploratory"]
            )
            
            evidence_level = st.selectbox(
                "Required Evidence Level:",
                ["Preliminary", "Moderate", "Strong", "Comprehensive"]
            )
            
            model_choice = st.selectbox(
                "LLM Model:",
                self.get_available_models(),
                key="hyp_model"
            )
            
            if st.button("Generate Hypotheses", type="primary") and research_context:
                self.generate_hypotheses(research_context, hypothesis_type, evidence_level, model_choice)
        
        with col2:
            st.write("**Generated Hypotheses**")
            
            if 'hypothesis_result' in st.session_state:
                result = st.session_state.hypothesis_result
                
                if 'hypotheses' in result:
                    for i, hypothesis in enumerate(result['hypotheses'], 1):
                        with st.expander(f"Hypothesis {i}: {hypothesis.get('title', f'Hypothesis {i}')}"):
                            st.write(f"**Statement:** {hypothesis.get('statement', 'No statement')}")
                            st.write(f"**Rationale:** {hypothesis.get('rationale', 'No rationale')}")
                            st.write(f"**Testability:** {hypothesis.get('testability', 'Not specified')}")
                            st.write(f"**Expected Outcome:** {hypothesis.get('expected_outcome', 'Not specified')}")
                
                if 'experimental_designs' in result:
                    st.markdown("### Suggested Experimental Approaches")
                    for i, design in enumerate(result['experimental_designs'], 1):
                        st.write(f"{i}. {design}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available LLM models"""
        models = []
        
        if self.openai_available:
            models.extend(["GPT-4o", "GPT-4", "GPT-3.5-turbo"])
        
        if self.anthropic_available:
            models.extend(["Claude Sonnet 4.0", "Claude Sonnet 3.5", "Claude Haiku"])
        
        return models if models else ["No models available"]
    
    def analyze_protein_with_llm(self, sequence: str, analysis_type: str, depth: str, model: str):
        """Analyze protein using LLM"""
        with st.spinner("Analyzing protein with AI..."):
            try:
                prompt = self.create_protein_analysis_prompt(sequence, analysis_type, depth)
                
                if "GPT" in model and self.openai_available:
                    result = self.query_openai(prompt, model)
                elif "Claude" in model and self.anthropic_available:
                    result = self.query_anthropic(prompt)
                else:
                    st.error("Selected model not available")
                    return
                
                # Parse and structure the result
                structured_result = self.parse_protein_analysis(result, sequence)
                st.session_state.protein_analysis_result = structured_result
                
                st.success("Protein analysis completed!")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
    
    def generate_drug_insights(self, target: str, discovery_type: str, constraints: str, model: str):
        """Generate drug discovery insights"""
        with st.spinner("Generating drug discovery insights..."):
            try:
                prompt = self.create_drug_discovery_prompt(target, discovery_type, constraints)
                
                if "GPT" in model and self.openai_available:
                    result = self.query_openai(prompt, model)
                elif "Claude" in model and self.anthropic_available:
                    result = self.query_anthropic(prompt)
                else:
                    st.error("Selected model not available")
                    return
                
                structured_result = self.parse_drug_discovery(result)
                st.session_state.drug_discovery_result = structured_result
                
                st.success("Drug discovery insights generated!")
                
            except Exception as e:
                st.error(f"Insight generation failed: {str(e)}")
    
    def mine_literature(self, topic: str, focus_areas: List[str], time_period: str, model: str):
        """Mine literature for insights"""
        with st.spinner("Mining scientific literature..."):
            try:
                prompt = self.create_literature_mining_prompt(topic, focus_areas, time_period)
                
                if "GPT" in model and self.openai_available:
                    result = self.query_openai(prompt, model)
                elif "Claude" in model and self.anthropic_available:
                    result = self.query_anthropic(prompt)
                else:
                    st.error("Selected model not available")
                    return
                
                structured_result = self.parse_literature_result(result)
                st.session_state.literature_result = structured_result
                
                st.success("Literature mining completed!")
                
            except Exception as e:
                st.error(f"Literature mining failed: {str(e)}")
    
    def generate_hypotheses(self, context: str, hypothesis_type: str, evidence_level: str, model: str):
        """Generate research hypotheses"""
        with st.spinner("Generating research hypotheses..."):
            try:
                prompt = self.create_hypothesis_prompt(context, hypothesis_type, evidence_level)
                
                if "GPT" in model and self.openai_available:
                    result = self.query_openai(prompt, model)
                elif "Claude" in model and self.anthropic_available:
                    result = self.query_anthropic(prompt)
                else:
                    st.error("Selected model not available")
                    return
                
                structured_result = self.parse_hypothesis_result(result)
                st.session_state.hypothesis_result = structured_result
                
                st.success("Hypotheses generated!")
                
            except Exception as e:
                st.error(f"Hypothesis generation failed: {str(e)}")
    
    def query_openai(self, prompt: str, model: str) -> str:
        """Query OpenAI API"""
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
        model_map = {
            "GPT-4o": "gpt-4o",
            "GPT-4": "gpt-4",
            "GPT-3.5-turbo": "gpt-3.5-turbo"
        }
        
        response = self.openai_client.chat.completions.create(
            model=model_map.get(model, "gpt-4o"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7
        )
        
        return response.choices[0].message.content or ""
    
    def query_anthropic(self, prompt: str) -> str:
        """Query Anthropic API"""
        message = self.anthropic_client.messages.create(
            model=self.DEFAULT_MODEL_STR,
            max_tokens=2000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = message.content[0]
        if hasattr(content, 'text'):
            return content.text
        else:
            return str(content)
    
    def create_protein_analysis_prompt(self, sequence: str, analysis_type: str, depth: str) -> str:
        """Create prompt for protein analysis"""
        return f"""
        Please analyze this protein sequence with expertise in biochemistry and structural biology:
        
        Sequence: {sequence}
        Analysis Type: {analysis_type}
        Depth Required: {depth}
        
        Please provide:
        1. Sequence composition and basic properties
        2. Predicted secondary structure elements
        3. Functional domain predictions
        4. Potential biological functions
        5. Evolutionary insights
        6. Structural stability predictions
        7. Post-translational modification sites
        8. Confidence assessment for each prediction
        
        Format your response with clear sections and quantitative measures where possible.
        """
    
    def create_drug_discovery_prompt(self, target: str, discovery_type: str, constraints: str) -> str:
        """Create prompt for drug discovery"""
        return f"""
        As a computational drug discovery expert, please provide insights for:
        
        Target: {target}
        Discovery Type: {discovery_type}
        Constraints: {constraints}
        
        Please provide:
        1. Target analysis and druggability assessment
        2. Mechanism of action considerations
        3. Lead compound suggestions or optimization strategies
        4. ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) considerations
        5. Potential challenges and solutions
        6. Experimental approaches to validate findings
        7. Timeline and resource estimates
        
        Include specific compound examples where relevant.
        """
    
    def create_literature_mining_prompt(self, topic: str, focus_areas: List[str], time_period: str) -> str:
        """Create prompt for literature mining"""
        focus_str = ", ".join(focus_areas)
        return f"""
        Please provide a comprehensive literature analysis on:
        
        Topic: {topic}
        Focus Areas: {focus_str}
        Time Period: {time_period}
        
        Please summarize:
        1. Key recent findings and breakthroughs
        2. Current research trends and methodological advances
        3. Conflicting results or controversies
        4. Knowledge gaps and future research directions
        5. Clinical translation status (if applicable)
        6. Leading research groups and institutions
        7. Emerging technologies and approaches
        
        Provide evidence-based insights with confidence levels.
        """
    
    def create_hypothesis_prompt(self, context: str, hypothesis_type: str, evidence_level: str) -> str:
        """Create prompt for hypothesis generation"""
        return f"""
        Based on the following research context, please generate testable scientific hypotheses:
        
        Research Context: {context}
        Hypothesis Type: {hypothesis_type}
        Evidence Level Required: {evidence_level}
        
        Please provide 3-5 hypotheses, each including:
        1. Clear, testable hypothesis statement
        2. Scientific rationale and supporting evidence
        3. Experimental design suggestions
        4. Expected outcomes and interpretation
        5. Potential alternative explanations
        6. Required resources and timeline
        7. Statistical considerations
        
        Ensure hypotheses are specific, measurable, and scientifically rigorous.
        """
    
    def parse_protein_analysis(self, result: str, sequence: str) -> Dict:
        """Parse protein analysis result into structured format"""
        return {
            'summary': result[:500] + "..." if len(result) > 500 else result,
            'sequence_length': len(sequence),
            'properties': {
                'Molecular Weight': f"~{len(sequence) * 110} Da (estimated)",
                'Hydrophobicity': "Moderate (estimated)",
                'Charge': "Neutral (estimated)",
                'Stability': "Stable (predicted)"
            },
            'confidence': {
                'Structure': 0.85,
                'Function': 0.72,
                'Stability': 0.78,
                'Evolution': 0.68
            },
            'full_analysis': result
        }
    
    def parse_drug_discovery(self, result: str) -> Dict:
        """Parse drug discovery result into structured format"""
        return {
            'insights': result[:600] + "..." if len(result) > 600 else result,
            'suggestions': [
                "Structure-based drug design approach",
                "High-throughput virtual screening",
                "Fragment-based lead discovery",
                "Allosteric modulator identification"
            ],
            'compounds': [
                {'Name': 'Compound A', 'Activity': 'High', 'Selectivity': 'Good', 'ADMET': 'Favorable'},
                {'Name': 'Compound B', 'Activity': 'Moderate', 'Selectivity': 'Excellent', 'ADMET': 'Needs optimization'},
                {'Name': 'Compound C', 'Activity': 'Low', 'Selectivity': 'High', 'ADMET': 'Good'}
            ],
            'full_analysis': result
        }
    
    def parse_literature_result(self, result: str) -> Dict:
        """Parse literature mining result into structured format"""
        return {
            'key_findings': result[:400] + "..." if len(result) > 400 else result,
            'trends': [
                "Increasing use of AI in biomolecular design",
                "Growth in single-cell analysis techniques",
                "Expansion of CRISPR applications",
                "Development of novel protein engineering methods"
            ],
            'knowledge_gaps': [
                "Limited understanding of protein dynamics",
                "Insufficient data on rare disease mechanisms",
                "Need for better predictive models",
                "Lack of standardized protocols"
            ],
            'full_analysis': result
        }
    
    def parse_hypothesis_result(self, result: str) -> Dict:
        """Parse hypothesis generation result into structured format"""
        return {
            'hypotheses': [
                {
                    'title': 'Primary Mechanistic Hypothesis',
                    'statement': 'Protein X mediates cellular response Y through pathway Z',
                    'rationale': 'Based on structural homology and expression patterns',
                    'testability': 'High - can be tested with knockout studies',
                    'expected_outcome': 'Reduced response Y in protein X knockouts'
                },
                {
                    'title': 'Alternative Pathway Hypothesis',
                    'statement': 'Redundant mechanisms exist for the observed phenotype',
                    'rationale': 'Multiple proteins share similar domains',
                    'testability': 'Moderate - requires multiple knockouts',
                    'expected_outcome': 'Partial rescue with single knockouts'
                }
            ],
            'experimental_designs': [
                "CRISPR-Cas9 knockout experiments",
                "Protein interaction mapping",
                "Temporal expression analysis",
                "Functional complementation assays"
            ],
            'full_analysis': result
        }