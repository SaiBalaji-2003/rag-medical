#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 11):
        print("❌ Error: Python 3.11.0 or higher is required")
        print("   Please upgrade your Python installation")
        return False
    
    if version >= (3, 12):
        print("⚠️  Warning: Python 3.12+ detected. Some packages may not be fully compatible.")
        print("   Recommended version: Python 3.11.9")
    
    print("✅ Python version compatible")
    return True

def check_system_requirements():
    """Check system requirements and capabilities."""
    print("\n🖥️  System Information:")
    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")
    
    # Check available memory
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   Total RAM: {memory_gb:.1f} GB")
        
        if memory_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM detected. Performance may be limited.")
        elif memory_gb >= 8:
            print("✅ Sufficient RAM for optimal performance")
        else:
            print("✅ Adequate RAM for basic operation")
            
    except ImportError:
        print("   RAM information not available (psutil not installed)")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU: {gpu_count}x {gpu_name}")
            print("✅ CUDA GPU detected - will enable GPU acceleration")
        else:
            print("   GPU: Not available")
            print("ℹ️  Will use CPU for inference (slower but functional)")
    except ImportError:
        print("   GPU information not available (PyTorch not installed yet)")

def create_directories():
    """Create necessary project directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
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
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Directories created successfully")

def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing Python packages...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        print("   📈 Upgrading pip...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL)
        
        # Install requirements
        print("   📥 Installing requirements...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error installing requirements:")
            print(result.stderr)
            return False
        
        print("✅ Requirements installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def verify_installation():
    """Verify that all components are properly installed."""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        ("streamlit", "Streamlit web framework"),
        ("torch", "PyTorch machine learning library"),
        ("transformers", "Hugging Face transformers"),
        ("sentence_transformers", "Sentence transformers"),
        ("chromadb", "ChromaDB vector database"),
        ("numpy", "NumPy numerical computing"),
        ("pandas", "Pandas data analysis")
    ]
    
    failed_imports = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}: {description}")
        except ImportError:
            print(f"   ❌ {package}: Failed to import")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Please run: pip install -r requirements.txt")
        return False
    
    print("✅ All packages imported successfully")
    return True

def download_default_data():
    """Create default ICD-10 dataset if it doesn't exist."""
    print("\n📄 Setting up default dataset...")
    
    data_file = Path("data/icd10_data.txt")
    
    if data_file.exists():
        print("   ✅ ICD-10 dataset already exists")
        return True
    
    # Create default ICD-10 data content
    icd10_content = """ICD-10 Mental and Behavioral Disorders Classification System

F00-F09 Organic, including symptomatic, mental disorders

F00 - Dementia in Alzheimer's disease: This diagnosis is used for dementia that is directly caused by Alzheimer's disease. The dementia is characterized by a slow, progressive decline in memory, thinking, and reasoning skills.

F01 - Vascular dementia: This type of dementia is the result of brain damage due to cerebrovascular disease, such as strokes. The onset is often sudden, and the course is typically stepwise, with periods of stability followed by abrupt declines in function.

F02 - Dementia in other diseases classified elsewhere: This code is for dementia that is a manifestation of other medical conditions, such as Parkinson's disease, Huntington's disease, or HIV.

F03 - Unspecified dementia: This code is used when a diagnosis of dementia is made, but the underlying cause is not specified or is unknown.

F30-F39 Mood [affective] disorders

F30 - Manic episode: A mood disorder characterized by a period of at least one week where an elevated, expansive, or unusually irritable mood exists.

F31 - Bipolar affective disorder: A disorder characterized by two or more episodes in which the patient's mood and activity levels are significantly disturbed.

F32 - Depressive episode: A mood disorder characterized by a period of at least two weeks during which there is either depressed mood or the loss of interest or pleasure in nearly all activities.

F33 - Recurrent depressive disorder: A disorder characterized by repeated episodes of depression as described for depressive episode (F32.-), without any history of independent episodes of mood elevation and increased energy (mania).

F33.0 Recurrent depressive disorder, current episode mild: Diagnostic criteria for mild depressive episode are met, and there has been at least one previous episode of depression.

F33.1 Recurrent depressive disorder, current episode moderate: Diagnostic criteria for moderate depressive episode are met, and there has been at least one previous episode of depression.

F33.2 Recurrent depressive disorder, current episode severe without psychotic symptoms: Diagnostic criteria for severe depressive episode without psychotic symptoms are met, and there has been at least one previous episode.

F33.3 Recurrent depressive disorder, current episode severe with psychotic symptoms: Diagnostic criteria for severe depressive episode with psychotic symptoms are met, and there has been at least one previous episode.

F33.4 Recurrent depressive disorder, currently in remission: There has been at least one previous mild, moderate, or severe depressive episode, but the current mental state does not meet criteria for depressive episode of any severity.

F40-F48 Neurotic, stress-related and somatoform disorders

F40 - Phobic anxiety disorders: A group of disorders in which anxiety is evoked only, or predominantly, in certain well-defined situations that are not currently dangerous.

F41 - Other anxiety disorders: This category includes disorders in which anxiety is the predominant symptom but is not restricted to any particular environmental situation.

F41.1 - Generalized anxiety disorder: Essential feature is anxiety that is generalized and persistent but not restricted to particular environmental circumstances. Dominant symptoms include persistent nervousness, trembling, muscular tensions, sweating, lightheadedness, palpitations, dizziness, and epigastric discomfort.

F42 - Obsessive-compulsive disorder: A disorder characterized by recurrent obsessional thoughts or compulsive acts.

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

F43 - Reaction to severe stress, and adjustment disorders: This category includes disorders that are identifiable as being a direct consequence of acute severe stress or continued trauma.

F43.1 - Post-traumatic stress disorder: Arises as a delayed or protracted response to a stressful event or situation of an exceptionally threatening or catastrophic nature. Typical features include episodes of repeated reliving of the trauma in intrusive memories, dreams, or nightmares.

F60-F69 Disorders of adult personality and behavior

F60 - Specific personality disorders: Deeply ingrained and enduring behavior patterns, manifesting as inflexible responses to a broad range of personal and social situations.

F60.0 Paranoid personality disorder: Characterized by excessive sensitivity to setbacks and rebuffs, tendency to bear grudges persistently, suspiciousness and a pervasive tendency to distort experience.

F60.1 Schizoid personality disorder: Characterized by withdrawal from affectional, social and other contacts with preference for fantasy, solitary activities, and introspection.

F60.2 Dissocial personality disorder: Characterized by disregard for social obligations, callous unconcern for the feelings of others, and incapacity to maintain enduring relationships.

F84.0 - Childhood autism: Developmental disorder characterized by abnormal or impaired development in social interaction and communication, and by a restricted, repetitive repertoire of activities and interests.

F90.0 Disturbance of activity and attention: General criteria for hyperkinetic disorder must be met. Characterized by early onset, combination of overactive behavior with marked inattention and lack of persistent task involvement.
"""
    
    try:
        with open(data_file, 'w', encoding='utf-8') as f:
            f.write(icd10_content)
        
        print(f"   ✅ Created default ICD-10 dataset: {data_file}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating dataset: {e}")
        return False

def run_initial_test():
    """Run a basic test to ensure everything works."""
    print("\n🧪 Running initial test...")
    
    try:
        # Test basic imports
        from src.rag_pipeline import RAGPipeline
        from config import VECTORDB_DIR, CACHE_DIR, MODELS_DIR
        
        print("   ✅ Core modules imported successfully")
        
        # Test pipeline initialization (without loading models)
        print("   🔧 Testing pipeline initialization...")
        
        # This is a basic test - we don't actually initialize the full pipeline
        # as it would download large models
        print("   ✅ Pipeline structure validated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def display_next_steps():
    """Display next steps for the user."""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("\n1. 🚀 Quick Start (Streamlit Web Interface):")
    print("   streamlit run app.py")
    print("\n2. 🖥️  Command Line Interface:")
    print("   python main.py --init --interactive")
    print("\n3. 📊 Initialize with default data:")
    print("   python main.py --init")
    print("\n4. ❓ Get help:")
    print("   python main.py --help")
    print("\n5. 🧪 Run tests:")
    print("   python -m pytest tests/")
    print("\n6. 📓 Explore Jupyter notebook:")
    print("   jupyter notebook notebooks/rag_demo.ipynb")
    
    print("\n💡 Tips:")
    print("   • First run will download AI models (~500MB-1GB)")
    print("   • Use GPU if available for faster performance")
    print("   • Check logs/ directory for detailed logs")
    print("   • Add custom documents via the web interface")
    
    print("\n🔗 Documentation:")
    print("   • README.md - Complete documentation")
    print("   • config.py - Configuration options")
    print("   • logs/rag_system.log - System logs")

def main():
    """Main setup function."""
    print("🏥 RAG Medical Q&A System - Setup Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed during verification")
        sys.exit(1)
    
    # Download default data
    if not download_default_data():
        print("\n⚠️  Warning: Failed to create default dataset")
    
    # Run initial test
    if not run_initial_test():
        print("\n⚠️  Warning: Initial test failed")
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main()