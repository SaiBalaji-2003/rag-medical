#!/usr/bin/env python3

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

from config import *
from src.rag_pipeline import RAGPipeline
from src.utils import setup_logging

def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG Medical Q&A System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --init                          # Initialize with default ICD-10 data
  python main.py --query "What is F33.4?"       # Ask a question
  python main.py --add-doc medical_data.txt     # Add custom document
  python main.py --interactive                  # Start interactive mode
  python main.py --benchmark                    # Run performance benchmark
        """
    )
    
    parser.add_argument(
        '--init', 
        action='store_true',
        help='Initialize the system with default ICD-10 dataset'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Ask a single question'
    )
    
    parser.add_argument(
        '--add-doc',
        type=str,
        help='Add a document to the knowledge base'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive query mode'
    )
    
    parser.add_argument(
        '--batch-queries',
        type=str,
        help='Process queries from a JSON file'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        help='Export knowledge base to specified file'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Display system statistics'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run system health check'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all data from the system'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of context chunks to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--no-rerank',
        action='store_true',
        help='Disable result reranking'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable query caching'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser

def initialize_system(pipeline):
    """Initialize the system with default ICD-10 data."""
    print("🏥 Initializing RAG Medical Q&A System...")
    print("=" * 60)
    
    try:
        default_data_path = DATA_DIR / "icd10_data.txt"
        
        if not default_data_path.exists():
            print(f"❌ Error: Default dataset not found at {default_data_path}")
            print("Please ensure the icd10_data.txt file exists in the data directory.")
            return False
        
        print(f"📁 Loading document: {default_data_path}")
        result = pipeline.initialize_from_documents([default_data_path])
        
        print(f"✅ Initialization completed!")
        print(f"📊 Statistics:")
        print(f"   • Documents processed: {result.get('documents_processed', 0)}")
        print(f"   • Total chunks created: {result.get('total_chunks', 0)}")
        print(f"   • Processing time: {result.get('processing_time_seconds', 0):.2f} seconds")
        print(f"   • Chunks per second: {result.get('chunks_per_second', 0):.1f}")
        
        if result.get('processing_errors'):
            print("⚠️  Processing errors:")
            for error in result['processing_errors']:
                print(f"   • {error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during initialization: {e}")
        return False

def process_single_query(pipeline, query, top_k, use_reranking, use_cache):
    """Process a single query and display results."""
    if not pipeline.is_initialized:
        print("❌ System not initialized. Please run with --init first.")
        return
    
    print(f"🔍 Processing query: {query}")
    print("-" * 60)
    
    start_time = datetime.now()
    
    result = pipeline.query(
        query,
        top_k=top_k,
        use_reranking=use_reranking,
        use_cache=use_cache
    )
    
    end_time = datetime.now()
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🎯 Answer:")
    print(f"   {result['answer']}")
    print()
    
    metadata = result.get('metadata', {})
    print(f"📊 Query Statistics:")
    print(f"   • Response time: {metadata.get('response_time_seconds', 0):.3f} seconds")
    print(f"   • Chunks retrieved: {metadata.get('chunks_retrieved', 0)}")
    print(f"   • Used reranking: {metadata.get('used_reranking', False)}")
    print(f"   • From cache: {metadata.get('from_cache', False)}")
    print()
    
    context = result.get('context', [])
    if context and len(context) > 0:
        print(f"📚 Top {len(context)} Context Chunks:")
        for i, chunk in enumerate(context):
            print(f"   {i+1}. Score: {chunk['similarity_score']:.4f}")
            print(f"      Content: {chunk['content'][:100]}...")
            if i < len(context) - 1:
                print()

def interactive_mode(pipeline, top_k, use_reranking, use_cache):
    """Start interactive query mode."""
    if not pipeline.is_initialized:
        print("❌ System not initialized. Please run with --init first.")
        return
    
    print("🏥 RAG Medical Q&A System - Interactive Mode")
    print("=" * 60)
    print("Enter your medical questions. Type 'quit', 'exit', or 'q' to end.")
    print("Type 'help' for available commands.")
    print()
    
    while True:
        try:
            query = input("❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print_help()
                continue
            
            if query.lower() == 'stats':
                display_stats(pipeline)
                continue
            
            if query.lower() == 'clear':
                print("\n" * 50)  # Clear screen
                continue
            
            if not query:
                continue
            
            print()
            process_single_query(pipeline, query, top_k, use_reranking, use_cache)
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def print_help():
    """Print help information for interactive mode."""
    print("""
📖 Available Commands:
   • Type any medical question to get an answer
   • 'stats' - Show system statistics
   • 'clear' - Clear the screen
   • 'help' - Show this help message
   • 'quit', 'exit', 'q' - Exit interactive mode

💡 Example Questions:
   • "What are the diagnostic criteria for OCD?"
   • "Give me the ICD-10 code for recurrent depression in remission"
   • "What is the difference between F32 and F33?"
   • "List all anxiety disorders in ICD-10"
""")

def display_stats(pipeline):
    """Display system statistics."""
    stats = pipeline.get_pipeline_stats()
    
    print("📊 System Statistics:")
    print(f"   • Status: {'Initialized' if pipeline.is_initialized else 'Not Initialized'}")
    print(f"   • Total queries: {stats.get('query_count', 0)}")
    print(f"   • Average response time: {stats.get('average_response_time_seconds', 0):.3f}s")
    
    vector_stats = stats.get('vector_store', {})
    print(f"   • Documents in database: {vector_stats.get('document_count', 0)}")
    
    embedding_info = stats.get('embedding_model', {})
    print(f"   • Embedding model: {embedding_info.get('model_name', 'Unknown')}")
    print(f"   • Embedding dimension: {embedding_info.get('embedding_dimension', 0)}")
    
    llm_info = stats.get('llm_model', {})
    print(f"   • Language model: {llm_info.get('model_name', 'Unknown')}")

def add_document(pipeline, doc_path):
    """Add a document to the knowledge base."""
    doc_file = Path(doc_path)
    
    if not doc_file.exists():
        print(f"❌ Error: Document not found: {doc_path}")
        return
    
    print(f"📁 Adding document: {doc_path}")
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        result = pipeline.add_document_content(content, doc_file.name)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Successfully added {result['chunks_added']} chunks")
            print(f"📊 Total collection size: {result['total_collection_size']}")
    
    except Exception as e:
        print(f"❌ Error adding document: {e}")

def batch_process_queries(pipeline, queries_file, top_k, use_reranking, use_cache):
    """Process multiple queries from a JSON file."""
    if not pipeline.is_initialized:
        print("❌ System not initialized. Please run with --init first.")
        return
    
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        if isinstance(queries_data, list):
            queries = queries_data
        elif isinstance(queries_data, dict) and 'queries' in queries_data:
            queries = queries_data['queries']
        else:
            print("❌ Invalid JSON format. Expected list of queries or dict with 'queries' key.")
            return
        
        print(f"🔄 Processing {len(queries)} queries from {queries_file}")
        print("=" * 60)
        
        results = []
        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: {query}")
            
            result = pipeline.query(
                query,
                top_k=top_k,
                use_reranking=use_reranking,
                use_cache=use_cache
            )
            
            results.append({
                'query': query,
                'answer': result.get('answer', ''),
                'metadata': result.get('metadata', {})
            })
            
            print(f"Answer: {result.get('answer', 'Error')}...")
            print()
        
        # Save results
        output_file = Path(queries_file).with_suffix('.results.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Results saved to {output_file}")
        
    except Exception as e:
        print(f"❌ Error processing batch queries: {e}")

def run_benchmark(pipeline):
    """Run performance benchmark."""
    if not pipeline.is_initialized:
        print("❌ System not initialized. Please run with --init first.")
        return
    
    print("🏃 Running Performance Benchmark...")
    print("=" * 60)
    
    benchmark_queries = [
        "What is the ICD-10 code for depression?",
        "Diagnostic criteria for OCD",
        "F33.4 definition",
        "Anxiety disorders classification",
        "Childhood autism criteria"
    ]
    
    total_time = 0
    results = []
    
    for i, query in enumerate(benchmark_queries, 1):
        print(f"Benchmark {i}/{len(benchmark_queries)}: {query}")
        
        start_time = datetime.now()
        result = pipeline.query(query, top_k=5)
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        total_time += response_time
        
        results.append({
            'query': query,
            'response_time': response_time,
            'success': 'error' not in result
        })
        
        print(f"Response time: {response_time:.3f}s")
        print()
    
    avg_time = total_time / len(benchmark_queries)
    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    
    print(f"📊 Benchmark Results:")
    print(f"   • Total queries: {len(benchmark_queries)}")
    print(f"   • Total time: {total_time:.3f}s")
    print(f"   • Average time: {avg_time:.3f}s")
    print(f"   • Success rate: {success_rate:.1f}%")
    print(f"   • Queries per second: {len(benchmark_queries)/total_time:.2f}")

def main():
    """Main function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(LOGGING_CONFIG["file"], log_level)
    
    # Initialize pipeline
    print("🚀 Starting RAG Medical Q&A System...")
    
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
        
        print("✅ Pipeline initialized successfully!")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process command line arguments
    if args.init:
        success = initialize_system(pipeline)
        if not success:
            sys.exit(1)
    
    if args.add_doc:
        add_document(pipeline, args.add_doc)
    
    if args.query:
        process_single_query(
            pipeline, 
            args.query, 
            args.top_k,
            not args.no_rerank,
            not args.no_cache
        )
    
    if args.batch_queries:
        batch_process_queries(
            pipeline,
            args.batch_queries,
            args.top_k,
            not args.no_rerank,
            not args.no_cache
        )
    
    if args.export:
        result = pipeline.export_data(Path(args.export))
        if result['success']:
            print(f"✅ Data exported to {result['export_path']}")
        else:
            print(f"❌ Export failed: {result['error']}")
    
    if args.stats:
        display_stats(pipeline)
    
    if args.health_check:
        print("🏥 Running Health Check...")
        health_status = pipeline.health_check()
        
        print(f"Overall Status: {health_status['overall_status']}")
        for component, status in health_status['components'].items():
            status_icon = "✅" if status['status'] == 'healthy' else "❌"
            print(f"{status_icon} {component}: {status['status']}")
            if status['status'] != 'healthy':
                print(f"   Error: {status.get('error', 'Unknown error')}")
    
    if args.benchmark:
        run_benchmark(pipeline)
    
    if args.clear:
        confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ")
        if confirm.lower() in ['yes', 'y']:
            pipeline.clear_all_data()
            print("🧹 All data cleared.")
        else:
            print("❌ Operation cancelled.")
    
    if args.interactive:
        interactive_mode(
            pipeline,
            args.top_k,
            not args.no_rerank,
            not args.no_cache
        )
    
    # If no specific action was requested, show help
    if not any([args.init, args.query, args.add_doc, args.interactive, 
                args.batch_queries, args.export, args.stats, 
                args.health_check, args.benchmark, args.clear]):
        parser.print_help()
        print("\n💡 Quick start: python main.py --init --interactive")

if __name__ == "__main__":
    main()