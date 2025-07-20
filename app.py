import streamlit as st
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import project modules
from config import *
from src.rag_pipeline import RAGPipeline
from src.utils import setup_logging, get_system_info, extract_icd_codes, validate_file_upload

# Configure Streamlit page
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging(LOGGING_CONFIG["file"], LOGGING_CONFIG["level"])
logger = logging.getLogger(__name__)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .context-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .sidebar-info {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        color: #c62828;
    }
    
    .success-box {
        background: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        color: #2e7d32;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 5px;
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with caching for better performance."""
    try:
        pipeline = RAGPipeline(
            vectordb_path=VECTORDB_DIR,
            cache_dir=CACHE_DIR,
            models_dir=MODELS_DIR,
            config={
                'chunking': CHUNK_CONFIG,
                'embedding': EMBEDDING_CONFIG,
                'vectordb': VECTORDB_CONFIG,
                'llm': LLM_CONFIG
            }
        )
        return pipeline
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None

def display_header():
    """Display the main application header."""
    st.markdown('<h1 class="main-header">🏥 RAG Medical Q&A System</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <p><strong>Welcome to the Medical Q&A System!</strong> This system uses Retrieval-Augmented Generation (RAG) 
        to answer questions about ICD-10 medical classifications and diagnostic criteria.</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar(pipeline):
    """Display the sidebar with system information and controls."""
    st.sidebar.title("🔧 System Control")
    
    # System status
    if pipeline:
        status = "🟢 Online" if pipeline.is_initialized else "🟡 Needs Initialization"
        st.sidebar.markdown(f"**Status:** {status}")
        
        # Pipeline stats
        if pipeline.is_initialized:
            stats = pipeline.get_pipeline_stats()
            
            st.sidebar.markdown("### 📊 Statistics")
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                st.metric("Documents", stats.get('vector_store', {}).get('document_count', 0))
                st.metric("Queries", stats.get('query_count', 0))
            
            with col2:
                avg_time = stats.get('average_response_time_seconds', 0)
                st.metric("Avg Response", f"{avg_time:.2f}s")
                
                collection_size = stats.get('vector_store', {}).get('document_count', 0)
                st.metric("Vector DB Size", collection_size)
    else:
        st.sidebar.error("❌ System Offline")
    
    # Configuration display
    st.sidebar.markdown("### ⚙️ Configuration")
    st.sidebar.markdown(f"""
    <div class="sidebar-info">
    <small>
    <strong>Chunking:</strong> {CHUNK_CONFIG['min_tokens']}-{CHUNK_CONFIG['max_tokens']} tokens<br>
    <strong>Embedding:</strong> {EMBEDDING_CONFIG['model_name']}<br>
    <strong>Vector Store:</strong> ChromaDB<br>
    <strong>LLM:</strong> {LLM_CONFIG['model_name']}
    </small>
    </div>
    """, unsafe_allow_html=True)
    
    # System info
    if st.sidebar.button("🖥️ System Info"):
        system_info = get_system_info()
        st.sidebar.json(system_info)

def document_management_tab(pipeline):
    """Handle document upload and management."""
    st.header("📁 Document Management")
    
    # Default document loading
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Load Default ICD-10 Dataset")
        if st.button("📥 Load ICD-10 Data", type="primary"):
            with st.spinner("Loading and processing ICD-10 dataset..."):
                try:
                    default_data_path = DATA_DIR / "icd10_data.txt"
                    if default_data_path.exists():
                        result = pipeline.initialize_from_documents([default_data_path])
                        
                        if 'processing_errors' in result and result['processing_errors']:
                            st.warning("Some processing errors occurred:")
                            for error in result['processing_errors']:
                                st.error(error)
                        else:
                            st.success("✅ ICD-10 dataset loaded successfully!")
                            
                        # Display processing stats
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Documents Processed", result.get('documents_processed', 0))
                        with col_b:
                            st.metric("Total Chunks", result.get('total_chunks', 0))
                        with col_c:
                            st.metric("Processing Time", f"{result.get('processing_time_seconds', 0):.1f}s")
                    else:
                        st.error("Default ICD-10 dataset not found!")
                        
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
    
    with col2:
        if pipeline.is_initialized:
            st.success("✅ System Ready")
        else:
            st.warning("⚠️ Load documents first")
    
    st.divider()
    
    # File upload section
    st.subheader("📤 Upload Custom Documents")
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt', 'md'],
        help="Upload additional medical documents to expand the knowledge base"
    )
    
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        
        # Validate file
        is_valid, error_msg = validate_file_upload(file_content, UI_CONFIG["max_file_size"])
        
        if not is_valid:
            st.error(f"File validation failed: {error_msg}")
        else:
            if st.button("🔄 Process Uploaded Document"):
                with st.spinner("Processing uploaded document..."):
                    try:
                        content_str = file_content.decode('utf-8')
                        result = pipeline.add_document_content(content_str, uploaded_file.name)
                        
                        if 'error' in result:
                            st.error(f"Processing failed: {result['error']}")
                        else:
                            st.success(f"✅ Added {result['chunks_added']} chunks from {uploaded_file.name}")
                            st.info(f"Total collection size: {result['total_collection_size']} documents")
                            
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
    
    # Collection management
    if pipeline.is_initialized:
        st.divider()
        st.subheader("🗄️ Collection Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Stats"):
                stats = pipeline.get_pipeline_stats()
                st.json(stats)
        
        with col2:
            if st.button("🧹 Clear Collection"):
                if st.confirm("Are you sure you want to clear all documents?"):
                    pipeline.clear_all_data()
                    st.success("Collection cleared!")
                    st.rerun()
        
        with col3:
            if st.button("💾 Export Data"):
                export_path = CACHE_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                result = pipeline.export_data(export_path)
                if result['success']:
                    st.success(f"Data exported to {result['export_path']}")
                else:
                    st.error(f"Export failed: {result['error']}")

def query_interface_tab(pipeline):
    """Main query interface for asking questions."""
    st.header("💬 Ask Questions")
    
    if not pipeline.is_initialized:
        st.warning("⚠️ Please load documents first in the Document Management tab.")
        return
    
    # Query input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your medical question:",
            height=100,
            placeholder="Example: What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?"
        )
    
    with col2:
        st.markdown("### ⚙️ Search Options")
        top_k = st.slider("Results to retrieve", 1, 20, 5)
        use_reranking = st.checkbox("Use reranking", value=True)
        use_cache = st.checkbox("Use cache", value=True)
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "Give me the correct coded classification for 'Recurrent depressive disorder, currently in remission'",
        "What are the diagnostic criteria for Obsessive-Compulsive Disorder (OCD)?",
        "What is the difference between F32 and F33 depression codes?",
        "List all personality disorders in ICD-10",
        "What are the criteria for childhood autism (F84.0)?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(sample_questions):
        col = cols[i % 2]
        if col.button(f"📝 {question[:50]}...", key=f"sample_{i}"):
            st.session_state.query_input = question
            st.rerun()
    
    # Process query
    if st.button("🔍 Get Answer", type="primary") or query:
        if query.strip():
            with st.spinner("Processing your question..."):
                start_time = datetime.now()
                
                result = pipeline.query(
                    query,
                    top_k=top_k,
                    use_reranking=use_reranking,
                    use_cache=use_cache
                )
                
                end_time = datetime.now()
                
                # Display results
                if 'error' in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Answer section
                    st.markdown("### 🎯 Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                        <h4>💡 Response:</h4>
                        <p>{result['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Extract and display ICD codes if found
                    icd_codes = extract_icd_codes(result['answer'])
                    if icd_codes:
                        st.markdown("### 🏷️ Identified ICD-10 Codes")
                        for code in icd_codes:
                            st.code(code, language="text")
                    
                    # Response metadata
                    metadata = result.get('metadata', {})
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Response Time", f"{metadata.get('response_time_seconds', 0):.2f}s")
                    with col2:
                        st.metric("Chunks Used", metadata.get('chunks_retrieved', 0))
                    with col3:
                        st.metric("Reranking", "Yes" if metadata.get('used_reranking') else "No")
                    with col4:
                        st.metric("Cached", "Yes" if metadata.get('from_cache') else "No")
                    
                    # Context section
                    st.markdown("### 📚 Retrieved Context")
                    context = result.get('context', [])
                    
                    if context:
                        for i, chunk in enumerate(context):
                            with st.expander(f"Context {i+1} (Score: {chunk['similarity_score']:.3f})"):
                                st.markdown(f"""
                                <div class="context-card">
                                    <p><strong>Content:</strong></p>
                                    <p>{chunk['content']}</p>
                                    <hr>
                                    <small>
                                    <strong>Similarity:</strong> {chunk['similarity_score']:.4f} | 
                                    <strong>Words:</strong> {chunk['word_count']} | 
                                    <strong>Characters:</strong> {chunk['char_count']}
                                    </small>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("No context retrieved for this query.")
        else:
            st.warning("Please enter a question.")

def analytics_tab(pipeline):
    """Display analytics and visualizations."""
    st.header("📈 Analytics & Performance")
    
    if not pipeline.is_initialized:
        st.warning("⚠️ No data available. Please load documents first.")
        return
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    
    # Overview metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Total Documents</p>
        </div>
        """.format(stats.get('vector_store', {}).get('document_count', 0)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Total Queries</p>
        </div>
        """.format(stats.get('query_count', 0)), unsafe_allow_html=True)
    
    with col3:
        avg_time = stats.get('average_response_time_seconds', 0)
        st.markdown("""
        <div class="metric-card">
            <h3>{:.2f}s</h3>
            <p>Avg Response Time</p>
        </div>
        """.format(avg_time), unsafe_allow_html=True)
    
    with col4:
        embedding_dim = stats.get('embedding_model', {}).get('embedding_dimension', 0)
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>Embedding Dimension</p>
        </div>
        """.format(embedding_dim), unsafe_allow_html=True)
    
    # Model information
    st.subheader("🤖 Model Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Embedding Model**")
        embedding_info = stats.get('embedding_model', {})
        st.json(embedding_info)
    
    with col2:
        st.markdown("**Language Model**")
        llm_info = stats.get('llm_model', {})
        st.json(llm_info)
    
    # Performance visualization
    st.subheader("📈 Performance Trends")
    
    # Create sample performance data for visualization
    if stats.get('query_count', 0) > 0:
        # Generate sample data based on actual stats
        sample_data = {
            'Query': [f"Query {i+1}" for i in range(min(10, stats.get('query_count', 0)))],
            'Response Time (s)': [avg_time + (i * 0.1) for i in range(min(10, stats.get('query_count', 0)))]
        }
        
        df = pd.DataFrame(sample_data)
        
        fig = px.line(df, x='Query', y='Response Time (s)', 
                     title='Response Time Trend',
                     markers=True)
        fig.update_layout(
            xaxis_title="Query Number",
            yaxis_title="Response Time (seconds)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # System health check
    st.subheader("🏥 System Health")
    if st.button("🔍 Run Health Check"):
        with st.spinner("Running health check..."):
            health_status = pipeline.health_check()
            
            if health_status['overall_status'] == 'healthy':
                st.success("✅ All systems healthy!")
            else:
                st.warning("⚠️ Some issues detected")
            
            # Display component status
            for component, status in health_status['components'].items():
                if status['status'] == 'healthy':
                    st.success(f"✅ {component}: Healthy")
                else:
                    st.error(f"❌ {component}: {status.get('error', 'Error')}")

def main():
    """Main application function."""
    # Display header
    display_header()
    
    # Initialize pipeline
    pipeline = initialize_rag_pipeline()
    
    # Display sidebar
    display_sidebar(pipeline)
    
    if pipeline is None:
        st.error("❌ System initialization failed. Please check the logs.")
        return
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📁 Document Management", "💬 Query Interface", "📈 Analytics"])
    
    with tab1:
        document_management_tab(pipeline)
    
    with tab2:
        query_interface_tab(pipeline)
    
    with tab3:
        analytics_tab(pipeline)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏥 RAG Medical Q&A System | Built with Streamlit, ChromaDB, and Transformers</p>
        <p><small>For educational and research purposes only. Not for clinical diagnosis.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()