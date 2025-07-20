#!/usr/bin/env python3
"""
RAG Medical Q&A System - One-Click Deployment Script
This script automates the complete setup and deployment of the RAG system.
"""

import os
import sys
import subprocess
import time
import platform
from pathlib import Path
import json

class RAGDeployment:
    def __init__(self):
        self.project_root = Path.cwd()
        self.python_executable = sys.executable
        self.deployment_log = []
        
    def log(self, message, level="INFO"):
        """Log deployment messages."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        self.deployment_log.append(log_entry)
    
    def run_command(self, command, description="", check=True):
        """Run a command and handle errors."""
        self.log(f"Running: {description or command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            if check and result.returncode != 0:
                self.log(f"Command failed: {command}", "ERROR")
                self.log(f"Error output: {result.stderr}", "ERROR")
                return False
            
            if result.stdout.strip():
                self.log(f"Output: {result.stdout.strip()}")
            
            return True
            
        except Exception as e:
            self.log(f"Exception running command: {e}", "ERROR")
            return False
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.log("🐍 Checking Python version...")
        
        version = sys.version_info
        self.log(f"Python version: {version.major}.{version.minor}.{version.micro}")
        
        if version < (3, 11):
            self.log("❌ Python 3.11+ required", "ERROR")
            return False
        
        if version >= (3, 12):
            self.log("⚠️  Python 3.12+ detected - some packages may have compatibility issues", "WARN")
        
        self.log("✅ Python version compatible")
        return True
    
    def create_project_structure(self):
        """Create the complete project directory structure."""
        self.log("📁 Creating project structure...")
        
        directories = [
            "src",
            "data",
            "data/uploads", 
            "vectordb",
            "cache",
            "models",
            "logs",
            "tests",
            "notebooks",
            "assets"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log(f"   Created: {directory}")
        
        self.log("✅ Project structure created")
        return True
    
    def create_requirements_file(self):
        """Create requirements.txt file."""
        self.log("📋 Creating requirements.txt...")
        
        requirements_content = """streamlit==1.28.1
torch==2.0.1
transformers==4.30.2
sentence-transformers==2.2.2
chromadb==0.4.15
numpy==1.24.3
pandas==2.0.3
tiktoken==0.5.1
scikit-learn==1.3.0
huggingface-hub==0.16.4
tokenizers==0.13.3
tqdm==4.65.0
requests==2.31.0
plotly==5.17.0
altair==5.1.2
watchdog==3.0.0
python-dotenv==1.0.0
langchain==0.0.350
langchain-community==0.0.6
openai==1.3.7
pillow==10.0.1
markdown==3.5.1
pytest==7.4.0
jupyter==1.0.0
psutil==5.9.5"""
        
        requirements_file = self.project_root / "requirements.txt"
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        
        self.log("✅ requirements.txt created")
        return True
    
    def install_dependencies(self):
        """Install Python dependencies."""
        self.log("📦 Installing dependencies...")
        
        # Upgrade pip first
        if not self.run_command(
            f'"{self.python_executable}" -m pip install --upgrade pip',
            "Upgrading pip"
        ):
            return False
        
        # Install requirements
        if not self.run_command(
            f'"{self.python_executable}" -m pip install -r requirements.txt',
            "Installing requirements"
        ):
            return False
        
        self.log("✅ Dependencies installed")
        return True
    
    def create_sample_data(self):
        """Create sample ICD-10 data file."""
        self.log("📄 Creating sample data...")
        
        sample_data = """ICD-10 Mental and Behavioral Disorders Classification System

F33 - Recurrent depressive disorder

F33.0 Recurrent depressive disorder, current episode mild: Diagnostic criteria for mild depressive episode are met, and there has been at least one previous episode of depression.

F33.1 Recurrent depressive disorder, current episode moderate: Diagnostic criteria for moderate depressive episode are met, and there has been at least one previous episode of depression.

F33.2 Recurrent depressive disorder, current episode severe without psychotic symptoms: Diagnostic criteria for severe depressive episode without psychotic symptoms are met, and there has been at least one previous episode.

F33.3 Recurrent depressive disorder, current episode severe with psychotic symptoms: Diagnostic criteria for severe depressive episode with psychotic symptoms are met, and there has been at least one previous episode.

F33.4 Recurrent depressive disorder, currently in remission: There has been at least one previous mild, moderate, or severe depressive episode, but the current mental state does not meet criteria for depressive episode of any severity.

F42 - Obsessive-compulsive disorder

Diagnostic criteria for Obsessive-Compulsive Disorder (OCD):
A. Presence of obsessions, compulsions, or both:

Obsessions are defined by (1) and (2):
1. Recurrent and persistent thoughts, urges, or images that are experienced as intrusive and unwanted
2. The individual attempts to ignore or suppress such thoughts, urges, or images, or to neutralize them with some other thought or action

Compulsions are defined by (1) and (2):
1. Repetitive behaviors or mental acts that the individual feels driven to perform in response to an obsession or according to rules that must be applied rigidly
2. The behaviors or mental acts are aimed at preventing or reducing anxiety or distress, or preventing some dreaded event or situation

B. The obsessions or compulsions are time-consuming (take more than 1 hour per day) or cause clinically significant distress or impairment in social, occupational, or other important areas of functioning.

C. The obsessive-compulsive symptoms are not attributable to the physiological effects of a substance or another medical condition.

D. The disturbance is not better explained by the symptoms of another mental disorder.

F41.1 - Generalized anxiety disorder: Essential feature is anxiety that is generalized and persistent but not restricted to particular environmental circumstances. Dominant symptoms include persistent nervousness, trembling, muscular tensions, sweating, lightheadedness, palpitations, dizziness, and epigastric discomfort.

F60.0 Paranoid personality disorder: Characterized by excessive sensitivity to setbacks and rebuffs, tendency to bear grudges persistently, suspiciousness and a pervasive tendency to distort experience.

F84.0 - Childhood autism: Developmental disorder characterized by abnormal or impaired development in social interaction and communication, and by a restricted, repetitive repertoire of activities and interests.

F90.0 Disturbance of activity and attention: General criteria for hyperkinetic disorder must be met. Characterized by early onset, combination of overactive behavior with marked inattention and lack of persistent task involvement."""
        
        data_file = self.project_root / "data" / "icd10_data.txt"
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(sample_data)
        
        self.log("✅ Sample data created")
        return True
    
    def create_minimal_source_files(self):
        """Create minimal source files to get the system running."""
        self.log("🔧 Creating minimal source files...")
        
        # Create __init__.py files
        init_files = [
            "src/__init__.py",
            "tests/__init__.py"
        ]
        
        for init_file in init_files:
            file_path = self.project_root / init_file
            file_path.touch()
            self.log(f"   Created: {init_file}")
        
        # Create minimal config.py
        config_content = '''import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORDB_DIR = PROJECT_ROOT / "vectordb"
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

for dir_path in [DATA_DIR, UPLOAD_DIR, VECTORDB_DIR, CACHE_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

CHUNK_CONFIG = {
    "min_tokens": 200,
    "max_tokens": 500,
    "overlap_tokens": 50,
    "chunk_strategy": "sentence_aware"
}

EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "batch_size": 32,
    "normalize_embeddings": True
}

VECTORDB_CONFIG = {
    "collection_name": "icd10_documents",
    "distance_metric": "cosine",
    "top_k_default": 5,
    "max_results": 20
}

LLM_CONFIG = {
    "model_name": "distilgpt2",
    "fallback_model": "gpt2",
    "max_tokens": 512,
    "temperature": 0.7
}

UI_CONFIG = {
    "page_title": "RAG Medical Q&A System",
    "page_icon": "🏥",
    "layout": "wide",
    "max_file_size": 10
}

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "rag_system.log"
}'''
        
        config_file = self.project_root / "config.py"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Create minimal main.py
        main_content = '''#!/usr/bin/env python3
import sys
from pathlib import Path

print("🏥 RAG Medical Q&A System")
print("System successfully deployed!")
print()
print("Next steps:")
print("1. Run: streamlit run app.py")
print("2. Or use CLI: python main.py --help")
print()
print("For full functionality, copy all source files from the artifacts.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--init":
        print("✅ System initialized (minimal version)")
        print("Please implement full RAG pipeline from artifacts")
    else:
        print("Usage: python main.py --init")
'''
        
        main_file = self.project_root / "main.py"
        with open(main_file, 'w') as f:
            f.write(main_content)
        
        # Create minimal app.py
        app_content = '''import streamlit as st

st.set_page_config(
    page_title="RAG Medical Q&A System",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 RAG Medical Q&A System")
st.success("✅ System successfully deployed!")

st.markdown("""
## 🚀 Quick Start

Your RAG system is now deployed! To enable full functionality:

1. **Copy Source Files**: Copy all source files from the provided artifacts
2. **Initialize System**: Run `python main.py --init`
3. **Start Querying**: Use this interface or CLI to ask questions

## 📁 File Structure Created

- ✅ Project directories
- ✅ Requirements file
- ✅ Sample ICD-10 data
- ✅ Configuration files
- ⏳ Core source files (copy from artifacts)

## 🔗 Resources

- **Implementation Guide**: Follow the complete guide
- **Sample Data**: ICD-10 mental health classifications loaded
- **Dependencies**: All Python packages installed

## 📋 Next Steps

1. Copy all source files from the artifacts
2. Run `python main.py --init` to initialize
3. Start asking medical questions!
""")

if st.button("🧪 Test Installation"):
    st.success("✅ Streamlit is working correctly!")
    st.info("📋 To enable full RAG functionality, implement the complete source code from artifacts")
'''
        
        app_file = self.project_root / "app.py"
        with open(app_file, 'w') as f:
            f.write(app_content)
        
        self.log("✅ Minimal source files created")
        return True
    
    def verify_installation(self):
        """Verify the installation is working."""
        self.log("🔍 Verifying installation...")
        
        # Test basic imports
        test_imports = [
            ("streamlit", "Streamlit web framework"),
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("sentence_transformers", "Sentence Transformers"),
            ("chromadb", "ChromaDB"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas")
        ]
        
        for package, description in test_imports:
            try:
                __import__(package)
                self.log(f"   ✅ {package}: {description}")
            except ImportError as e:
                self.log(f"   ❌ {package}: Failed to import ({e})", "ERROR")
                return False
        
        # Test file structure
        required_files = [
            "requirements.txt",
            "config.py", 
            "main.py",
            "app.py",
            "data/icd10_data.txt"
        ]
        
        for file_path in required_files:
            file_full_path = self.project_root / file_path
            if file_full_path.exists():
                self.log(f"   ✅ {file_path}: Found")
            else:
                self.log(f"   ❌ {file_path}: Missing", "ERROR")
                return False
        
        self.log("✅ Installation verified")
        return True
    
    def save_deployment_log(self):
        """Save deployment log for debugging."""
        log_file = self.project_root / "deployment.log"
        with open(log_file, 'w') as f:
            f.write("\n".join(self.deployment_log))
        self.log(f"💾 Deployment log saved to: {log_file}")
    
    def show_next_steps(self):
        """Display next steps to the user."""
        self.log("🎉 Deployment completed successfully!")
        
        next_steps = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║                              🎉 DEPLOYMENT SUCCESSFUL! 🎉                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

📋 NEXT STEPS:

1. 📂 COPY SOURCE FILES:
   Copy all the source files from the artifacts provided:
   - src/document_processor.py
   - src/embeddings.py  
   - src/vector_store.py
   - src/llm_interface.py
   - src/rag_pipeline.py
   - src/utils.py
   - Complete app.py and main.py files

2. 🚀 START THE SYSTEM:
   
   Web Interface (Recommended):
   └─ streamlit run app.py
   
   Command Line:
   └─ python main.py --init
   └─ python main.py --interactive

3. 🧪 TEST THE SYSTEM:
   └─ python main.py --query "What is F33.4?"
   └─ python main.py --health-check

4. 📚 EXPLORE:
   └─ jupyter notebook notebooks/rag_demo.ipynb

💡 QUICK START COMMANDS:
   
   # Start web interface
   streamlit run app.py
   
   # Initialize with sample data  
   python main.py --init
   
   # Ask a question
   python main.py --query "What are the diagnostic criteria for OCD?"

🔗 RESOURCES:
   • README.md - Complete documentation
   • IMPLEMENTATION_GUIDE.md - Step-by-step setup  
   • deployment.log - This deployment log
   • data/icd10_data.txt - Sample medical data

⚠️  IMPORTANT: 
   This is a minimal deployment. Copy all source files from artifacts for full functionality.

🏥 Your RAG Medical Q&A System is ready!
"""
        print(next_steps)
    
    def deploy(self):
        """Main deployment function."""
        start_time = time.time()
        
        print("🚀 Starting RAG Medical Q&A System Deployment")
        print("=" * 80)
        
        deployment_steps = [
            ("Python Version Check", self.check_python_version),
            ("Project Structure", self.create_project_structure),
            ("Requirements File", self.create_requirements_file),
            ("Dependencies", self.install_dependencies),
            ("Sample Data", self.create_sample_data),
            ("Source Files", self.create_minimal_source_files),
            ("Verification", self.verify_installation)
        ]
        
        for step_name, step_function in deployment_steps:
            self.log(f"📋 Step: {step_name}")
            if not step_function():
                self.log(f"❌ Deployment failed at step: {step_name}", "ERROR")
                self.save_deployment_log()
                return False
            self.log(f"✅ Step completed: {step_name}")
            print()
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.log(f"⏱️  Total deployment time: {duration:.1f} seconds")
        self.save_deployment_log()
        self.show_next_steps()
        
        return True

def main():
    """Main function."""
    deployment = RAGDeployment()
    
    try:
        success = deployment.deploy()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Deployment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()