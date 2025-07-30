import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import os
import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseInterfaceComponent:
    """Comprehensive database interface for biomolecular platform"""
    
    def __init__(self):
        self.db_params = {
            'host': os.getenv('PGHOST'),
            'port': os.getenv('PGPORT'),
            'database': os.getenv('PGDATABASE'),
            'user': os.getenv('PGUSER'),
            'password': os.getenv('PGPASSWORD')
        }
    
    def get_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_params)
        except Exception as e:
            st.error(f"Database connection failed: {str(e)}")
            return None
    
    def render(self):
        """Render database interface"""
        st.header("🗄️ Biomolecular Database")
        st.markdown("Comprehensive data management and analytics platform")
        
        # Database status
        self.show_database_status()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Explorer", 
            "Analytics Dashboard", 
            "Search & Query", 
            "Data Management",
            "Export & Import"
        ])
        
        with tab1:
            self.render_data_explorer()
        
        with tab2:
            self.render_analytics_dashboard()
        
        with tab3:
            self.render_search_query()
        
        with tab4:
            self.render_data_management()
        
        with tab5:
            self.render_export_import()
    
    def show_database_status(self):
        """Show database connection status and basic stats"""
        conn = self.get_connection()
        if not conn:
            st.error("Database connection failed. Please check configuration.")
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get table counts
                tables = ['proteins', 'compounds', 'experiments', 'protein_analyses']
                counts = {}
                
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = cur.fetchone()
                    counts[table] = result['count'] if result else 0
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Proteins", counts.get('proteins', 0))
                
                with col2:
                    st.metric("Compounds", counts.get('compounds', 0))
                
                with col3:
                    st.metric("Experiments", counts.get('experiments', 0))
                
                with col4:
                    st.metric("Analyses", counts.get('protein_analyses', 0))
                
        except Exception as e:
            st.error(f"Error getting database stats: {str(e)}")
        finally:
            conn.close()
    
    def render_data_explorer(self):
        """Render data exploration interface"""
        st.subheader("📊 Data Explorer")
        
        # Table selector
        table_choice = st.selectbox(
            "Select Data Table:",
            ["proteins", "compounds", "experiments", "protein_analyses", "protein_compound_interactions"]
        )
        
        # Load and display data
        data = self.load_table_data(table_choice)
        
        if data is not None and not data.empty:
            # Data filtering
            st.write("**Filter Data:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Text search
                search_term = st.text_input("Search:", key=f"search_{table_choice}")
                
            with col2:
                # Row limit
                row_limit = st.slider("Max rows:", 10, 1000, 100, key=f"limit_{table_choice}")
            
            # Apply filters
            filtered_data = data.head(row_limit)
            
            if search_term:
                # Search across all text columns
                text_columns = filtered_data.select_dtypes(include=['object']).columns
                mask = filtered_data[text_columns].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                filtered_data = filtered_data[mask]
            
            # Display data
            st.dataframe(filtered_data, use_container_width=True, height=400)
            
            # Data summary
            st.write("**Data Summary:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"Total rows: {len(data)}")
            with col2:
                st.write(f"Displayed rows: {len(filtered_data)}")
            with col3:
                st.write(f"Columns: {len(data.columns)}")
        
        else:
            st.info(f"No data found in {table_choice} table.")
    
    def render_analytics_dashboard(self):
        """Render analytics dashboard"""
        st.subheader("📈 Analytics Dashboard")
        
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Experiment status distribution
                cur.execute("""
                    SELECT status, COUNT(*) as count 
                    FROM experiments 
                    GROUP BY status
                """)
                exp_status = cur.fetchall()
                
                if exp_status:
                    df_status = pd.DataFrame(exp_status)
                    
                    fig_pie = px.pie(
                        df_status, 
                        values='count', 
                        names='status',
                        title="Experiment Status Distribution"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Protein length distribution
                cur.execute("""
                    SELECT length, organism, COUNT(*) as count
                    FROM proteins 
                    WHERE length IS NOT NULL
                    GROUP BY length, organism
                    ORDER BY length
                """)
                protein_lengths = cur.fetchall()
                
                if protein_lengths:
                    df_lengths = pd.DataFrame(protein_lengths)
                    
                    fig_hist = px.histogram(
                        df_lengths, 
                        x='length', 
                        weights='count',
                        title="Protein Length Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Recent activity timeline
                cur.execute("""
                    SELECT DATE(created_at) as date, COUNT(*) as experiments
                    FROM experiments 
                    WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """)
                recent_activity = cur.fetchall()
                
                if recent_activity:
                    df_activity = pd.DataFrame(recent_activity)
                    
                    fig_line = px.line(
                        df_activity, 
                        x='date', 
                        y='experiments',
                        title="Experiment Activity (Last 30 Days)"
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                
                # Organism distribution
                cur.execute("""
                    SELECT organism, COUNT(*) as count
                    FROM proteins 
                    WHERE organism IS NOT NULL
                    GROUP BY organism
                    ORDER BY count DESC
                    LIMIT 10
                """)
                organisms = cur.fetchall()
                
                if organisms:
                    df_organisms = pd.DataFrame(organisms)
                    
                    fig_bar = px.bar(
                        df_organisms, 
                        x='organism', 
                        y='count',
                        title="Top 10 Organisms by Protein Count"
                    )
                    fig_bar.update_xaxis(tickangle=45)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")
        finally:
            conn.close()
    
    def render_search_query(self):
        """Render search and query interface"""
        st.subheader("🔍 Search & Custom Queries")
        
        # Quick search
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input(
                "Quick Search:",
                placeholder="Search proteins, compounds, experiments..."
            )
        
        with col2:
            search_type = st.selectbox(
                "Search In:",
                ["All", "Proteins", "Compounds", "Experiments"]
            )
        
        if st.button("Search") and search_query:
            results = self.perform_search(search_query, search_type)
            
            if results:
                st.write(f"Found {len(results)} results:")
                
                for result in results:
                    with st.expander(f"{result['type'].title()}: {result['name']}"):
                        for key, value in result.items():
                            if key not in ['type', 'name']:
                                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.info("No results found.")
        
        # Custom SQL query interface
        st.write("---")
        st.write("**Custom SQL Query:**")
        
        sql_query = st.text_area(
            "Enter SQL query:",
            height=100,
            placeholder="SELECT name, organism FROM proteins WHERE length > 100 LIMIT 10;"
        )
        
        if st.button("Execute Query") and sql_query:
            try:
                result_df = self.execute_custom_query(sql_query)
                if result_df is not None:
                    st.dataframe(result_df, use_container_width=True)
                else:
                    st.success("Query executed successfully (no results returned)")
            except Exception as e:
                st.error(f"Query error: {str(e)}")
    
    def render_data_management(self):
        """Render data management interface"""
        st.subheader("⚙️ Data Management")
        
        # Add new data
        data_type = st.selectbox(
            "Add New Data:",
            ["Protein", "Compound", "Experiment"]
        )
        
        if data_type == "Protein":
            self.render_add_protein_form()
        elif data_type == "Compound":
            self.render_add_compound_form()
        elif data_type == "Experiment":
            self.render_add_experiment_form()
        
        st.write("---")
        
        # Database maintenance
        st.write("**Database Maintenance:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Vacuum Database"):
                self.vacuum_database()
        
        with col2:
            if st.button("Update Statistics"):
                self.update_statistics()
        
        with col3:
            if st.button("Check Integrity"):
                self.check_integrity()
    
    def render_add_protein_form(self):
        """Render form to add new protein"""
        with st.form("add_protein"):
            name = st.text_input("Protein Name*")
            sequence = st.text_area("Sequence*", height=100)
            organism = st.text_input("Organism")
            uniprot_id = st.text_input("UniProt ID")
            description = st.text_area("Description")
            
            submitted = st.form_submit_button("Add Protein")
            
            if submitted and name and sequence:
                try:
                    conn = self.get_connection()
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO proteins (name, sequence, organism, uniprot_id, description, molecular_weight, length)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (name, sequence, organism, uniprot_id, description, len(sequence) * 110, len(sequence)))
                        conn.commit()
                    conn.close()
                    st.success(f"Protein '{name}' added successfully!")
                except Exception as e:
                    st.error(f"Error adding protein: {str(e)}")
    
    def render_add_compound_form(self):
        """Render form to add new compound"""
        with st.form("add_compound"):
            name = st.text_input("Compound Name*")
            smiles = st.text_input("SMILES")
            molecular_formula = st.text_input("Molecular Formula")
            molecular_weight = st.number_input("Molecular Weight", min_value=0.0)
            compound_type = st.selectbox("Type", ["drug", "metabolite", "natural_product", "synthetic"])
            
            submitted = st.form_submit_button("Add Compound")
            
            if submitted and name:
                try:
                    conn = self.get_connection()
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO compounds (name, smiles, molecular_formula, molecular_weight, compound_type)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (name, smiles, molecular_formula, molecular_weight, compound_type))
                        conn.commit()
                    conn.close()
                    st.success(f"Compound '{name}' added successfully!")
                except Exception as e:
                    st.error(f"Error adding compound: {str(e)}")
    
    def render_add_experiment_form(self):
        """Render form to add new experiment"""
        with st.form("add_experiment"):
            name = st.text_input("Experiment Name*")
            experiment_type = st.selectbox("Type", ["protein_analysis", "drug_discovery", "federated_learning", "structure_prediction"])
            description = st.text_area("Description")
            created_by = st.text_input("Created By")
            institution = st.text_input("Institution")
            
            submitted = st.form_submit_button("Add Experiment")
            
            if submitted and name:
                try:
                    conn = self.get_connection()
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO experiments (name, experiment_type, description, created_by, institution, start_time)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (name, experiment_type, description, created_by, institution, datetime.now()))
                        conn.commit()
                    conn.close()
                    st.success(f"Experiment '{name}' added successfully!")
                except Exception as e:
                    st.error(f"Error adding experiment: {str(e)}")
    
    def render_export_import(self):
        """Render export/import interface"""
        st.subheader("📤 Export & Import")
        
        # Export data
        st.write("**Export Data:**")
        
        export_table = st.selectbox(
            "Select table to export:",
            ["proteins", "compounds", "experiments", "protein_analyses"]
        )
        
        export_format = st.selectbox("Format:", ["CSV", "JSON"])
        
        if st.button("Export Data"):
            data = self.load_table_data(export_table)
            if data is not None:
                if export_format == "CSV":
                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{export_table}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:  # JSON
                    json_data = data.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"{export_table}_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
        
        st.write("---")
        
        # Import data
        st.write("**Import Data:**")
        
        uploaded_file = st.file_uploader(
            "Choose file to import:",
            type=['csv', 'json'],
            help="Upload CSV or JSON file to import data"
        )
        
        if uploaded_file is not None:
            import_table = st.selectbox(
                "Target table:",
                ["proteins", "compounds", "experiments"],
                key="import_table"
            )
            
            if st.button("Import Data"):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        import_data = pd.read_csv(uploaded_file)
                    else:  # JSON
                        import_data = pd.read_json(uploaded_file)
                    
                    st.write("Preview of imported data:")
                    st.dataframe(import_data.head())
                    
                    # Here you would implement the actual import logic
                    st.success(f"Ready to import {len(import_data)} rows to {import_table}")
                    
                except Exception as e:
                    st.error(f"Import error: {str(e)}")
    
    def load_table_data(self, table_name: str) -> pd.DataFrame:
        """Load data from specified table"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            query = f"SELECT * FROM {table_name} ORDER BY created_at DESC LIMIT 1000"
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            st.error(f"Error loading data from {table_name}: {str(e)}")
            return None
        finally:
            conn.close()
    
    def perform_search(self, query: str, search_type: str) -> List[Dict]:
        """Perform search across database"""
        conn = self.get_connection()
        if not conn:
            return []
        
        results = []
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if search_type in ["All", "Proteins"]:
                    cur.execute("""
                        SELECT 'protein' as type, id, name, organism, description
                        FROM proteins 
                        WHERE name ILIKE %s OR organism ILIKE %s OR description ILIKE %s
                        LIMIT 20
                    """, (f"%{query}%", f"%{query}%", f"%{query}%"))
                    results.extend(cur.fetchall())
                
                if search_type in ["All", "Compounds"]:
                    cur.execute("""
                        SELECT 'compound' as type, id, name, molecular_formula, compound_type
                        FROM compounds 
                        WHERE name ILIKE %s OR molecular_formula ILIKE %s
                        LIMIT 20
                    """, (f"%{query}%", f"%{query}%"))
                    results.extend(cur.fetchall())
                
                if search_type in ["All", "Experiments"]:
                    cur.execute("""
                        SELECT 'experiment' as type, id, name, experiment_type, description, status
                        FROM experiments 
                        WHERE name ILIKE %s OR description ILIKE %s
                        LIMIT 20
                    """, (f"%{query}%", f"%{query}%"))
                    results.extend(cur.fetchall())
                    
        except Exception as e:
            st.error(f"Search error: {str(e)}")
        finally:
            conn.close()
        
        return [dict(result) for result in results]
    
    def execute_custom_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query"""
        conn = self.get_connection()
        if not conn:
            return None
        
        try:
            df = pd.read_sql_query(query, conn)
            return df
        except Exception as e:
            raise e
        finally:
            conn.close()
    
    def vacuum_database(self):
        """Vacuum database for maintenance"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("VACUUM ANALYZE")
            st.success("Database vacuum completed successfully!")
        except Exception as e:
            st.error(f"Vacuum error: {str(e)}")
        finally:
            conn.close()
    
    def update_statistics(self):
        """Update database statistics"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor() as cur:
                cur.execute("ANALYZE")
                conn.commit()
            st.success("Database statistics updated successfully!")
        except Exception as e:
            st.error(f"Statistics update error: {str(e)}")
        finally:
            conn.close()
    
    def check_integrity(self):
        """Check database integrity"""
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check for orphaned records
                cur.execute("""
                    SELECT COUNT(*) as orphaned_analyses
                    FROM protein_analyses pa
                    LEFT JOIN proteins p ON pa.protein_id = p.id
                    WHERE p.id IS NULL
                """)
                
                result = cur.fetchone()
                orphaned = result['orphaned_analyses'] if result else 0
                
                if orphaned == 0:
                    st.success("Database integrity check passed - no issues found!")
                else:
                    st.warning(f"Found {orphaned} orphaned analysis records")
                    
        except Exception as e:
            st.error(f"Integrity check error: {str(e)}")
        finally:
            conn.close()
