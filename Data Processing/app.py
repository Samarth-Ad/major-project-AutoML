import streamlit as st
import pandas as pd
import os
import json
import time
from pathlib import Path
import sys

# Add current directory to path so we can import local modules
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from orchestrator.master_agent import MasterAgent
    from utils.logger import PipelineLogger
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure you are running from the project root.")

# Page Config
st.set_page_config(
    page_title="Agentic ML Pipeline Builder",
    page_icon="🤖",
    layout="wide",
)

# Sidebar - Configuration
st.sidebar.title("⚙️ Configuration")
backend = st.sidebar.selectbox("LLM Backend", ["ollama", "anthropic"], index=0)

# Default models
default_model = "gpt-oss:120b-cloud" if backend == "ollama" else "claude-sonnet-4-20250514"
model = st.sidebar.text_input("Model Name", value=default_model)
api_key = st.sidebar.text_input("API Key (if required)", type="password")

st.sidebar.markdown("---")
st.sidebar.info("This tool designs and executes an adaptive ML pipeline based on your data and instructions.")

# Main UI
st.title("🤖 Agentic ML Pipeline Builder")
st.markdown("""
Upload your dataset and provide a prompt to guide the pipeline construction. 
The system will automatically analyze your data and design the best steps.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Input Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write(f"Preview ({df_preview.shape[0]} rows x {df_preview.shape[1]} columns):")
            st.dataframe(df_preview.head(5))
            
            # Save temporary file
            temp_data_dir = Path("data")
            temp_data_dir.mkdir(exist_ok=True)
            temp_data_path = temp_data_dir / "temp_upload.csv"
            df_preview.to_csv(temp_data_path, index=False)
            st.session_state['data_path'] = str(temp_data_path)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

with col2:
    st.subheader("✍️ User Intent")
    user_prompt = st.text_area("What should the pipeline focus on?", 
                              placeholder="e.g., 'Handle missing values and optimize for explainability'",
                              height=150)
    
    run_button = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

# Execution and Results
if run_button:
    if uploaded_file is None:
        st.error("Please upload a dataset first!")
    else:
        # Set environment variables for the agent
        os.environ["LLM_BACKEND"] = backend
        if backend == "ollama":
            os.environ["OLLAMA_MODEL"] = model
        else:
            os.environ["ANTHROPIC_MODEL"] = model
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            
        st.divider()
        st.subheader("⛓️ Execution Progress")
        
        status_container = st.status("Initializing MasterAgent...", expanded=True)
        
        try:
            master = MasterAgent(
                api_key=api_key,
                llm_model=model,
                config_path="config/pipeline.yaml"
            )
            
            # Use a placeholder for live code display
            code_placeholder = st.empty()
            
            with status_container:
                st.write("🚀 Pipeline starting...")
                
                # Setup a loop to check for code updates while master.run is in another thread?
                # Actually, MasterAgent.run is blocking. For real-time in Streamlit without 
                # threading, we'd need to modify MasterAgent to yield or use a callback.
                # Since we want to display code, let's add a "Current Code" section below progress.
                
                result = master.run(
                    pipeline_config="config/pipeline.yaml",
                    initial_data=st.session_state['data_path'],
                    user_prompt=user_prompt
                )
            
            # After run, we can still show the final generated script
            st.divider()
            st.subheader("📝 Generated Pipeline Code")
            final_code_path = Path("generated_code/pipeline_script.py")
            if final_code_path.exists():
                st.code(final_code_path.read_text(encoding="utf-8"), language="python")
            
            if result.success:
                st.success("✅ Pipeline Execution Successful!")
                
                # Show Results in Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🧹 Cleaned Data", "📝 Report", "📜 Logs"])
                
                with tab1:
                    metrics_path = Path("outputs/metrics.json")
                    if metrics_path.exists():
                        with open(metrics_path, encoding="utf-8") as f:
                            metrics = json.load(f)
                        st.json(metrics)
                    else:
                        st.info("No metrics file found. Some steps might have been skipped.")
                        
                with tab2:
                    cleaned_path = Path("outputs/cleaned_data.csv")
                    if cleaned_path.exists():
                        df_cleaned = pd.read_csv(cleaned_path, encoding="utf-8")
                        st.write(f"Showing first 100 rows of {df_cleaned.shape[0]} total.")
                        st.dataframe(df_cleaned.head(100))
                        st.download_button("Download Cleaned CSV", 
                                         data=df_cleaned.to_csv(index=False),
                                         file_name="cleaned_data.csv",
                                         mime="text/csv")
                    else:
                        st.info("No cleaned data file found.")
                        
                with tab3:
                    report_path = Path("outputs/report.md")
                    if report_path.exists():
                        st.markdown(report_path.read_text(encoding="utf-8"))
                    else:
                        st.info("No report file found.")
                        
                with tab4:
                    log_path = Path("logs/pipeline.log")
                    if log_path.exists():
                        all_logs = log_path.read_text(encoding="utf-8")
                        # Filter for important events
                        important_keywords = ["INFO", "ERROR", "WARNING", "STEP START", "STEP END", "SUCCESS", "FAILED", "Sanitiser"]
                        filtered_lines = [
                            line for line in all_logs.splitlines() 
                            if any(kw in line for kw in important_keywords)
                        ]
                        # Show last 500 important lines
                        display_logs = "\n".join(filtered_lines[-500:])
                        st.text_area("Important Pipeline Logs", display_logs, height=400)
                    else:
                        st.warning("Log file not found.")
            else:
                st.error("❌ Pipeline Execution Failed. Check the 'Logs' tab for details.")
                
        except Exception as e:
            st.error(f"Fatal error during execution: {e}")
            st.exception(e)

# Footer
st.markdown("---")
st.caption("Agentic ML Pipeline Builder — Powered by LangChain and LLMs")
