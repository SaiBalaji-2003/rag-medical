import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .llm_interface import LLMInterface
from .utils import create_cache_key, save_cache, load_cache, format_chunks_for_display

class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline for medical Q&A.
    
    Features:
    - End-to-end document processing and indexing
    - Intelligent query processing with caching
    - Advanced retrieval with reranking
    - Context-aware answer generation
    - Performance monitoring and logging
    """
    
    def __init__(self, 
                 vectordb_path: Path,
                 cache_dir: Optional[Path] = None,
                 models_dir: Optional[Path] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAG pipeline with all components.
        
        Args:
            vectordb_path: Path for vector database storage
            cache_dir: Directory for query caching
            models_dir: Directory for model caching
            config: Configuration dictionary for all components
        """
        self.vectordb_path = vectordb_path
        self.cache_dir = cache_dir
        self.models_dir = models_dir
        self.config = config or {}
        
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0
        
        self.logger.info("RAGPipeline initialized successfully")
    
    def _initialize_components(self) -> None:
        """
        Initialize all pipeline components with configuration.
        """
        try:
            # Document processor
            chunk_config = self.config.get('chunking', {})
            self.doc_processor = DocumentProcessor(
                min_tokens=chunk_config.get('min_tokens', 200),
                max_tokens=chunk_config.get('max_tokens', 500),
                overlap_tokens=chunk_config.get('overlap_tokens', 50),
                chunk_strategy=chunk_config.get('chunk_strategy', 'sentence_aware')
            )
            
            # Embedding generator
            embedding_config = self.config.get('embedding', {})
            self.embedding_generator = EmbeddingGenerator(
                model_name=embedding_config.get('model_name', 'all-MiniLM-L6-v2'),
                cache_folder=self.models_dir,
                batch_size=embedding_config.get('batch_size', 32),
                normalize_embeddings=embedding_config.get('normalize_embeddings', True)
            )
            
            # Vector store
            vectordb_config = self.config.get('vectordb', {})
            self.vector_store = VectorStore(
                db_path=self.vectordb_path,
                collection_name=vectordb_config.get('collection_name', 'documents'),
                distance_metric=vectordb_config.get('distance_metric', 'cosine')
            )
            
            # LLM interface
            llm_config = self.config.get('llm', {})
            self.llm = LLMInterface(
                model_name=llm_config.get('model_name', 'microsoft/DialoGPT-medium'),
                cache_dir=self.models_dir,
                max_tokens=llm_config.get('max_tokens', 512),
                temperature=llm_config.get('temperature', 0.7)
            )
            
            self.logger.info("All pipeline components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise RuntimeError(f"Pipeline initialization failed: {e}")
    
    def initialize_from_documents(self, document_paths: List[Path]) -> Dict[str, Any]:
        """
        Initialize the pipeline by processing and indexing documents.
        
        Args:
            document_paths: List of paths to documents to process
            
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting document processing for {len(document_paths)} documents")
            
            all_chunks = []
            processing_stats = {
                'documents_processed': 0,
                'total_chunks': 0,
                'processing_errors': [],
                'start_time': start_time.isoformat()
            }
            
            # Process each document
            for doc_path in document_paths:
                try:
                    self.logger.info(f"Processing document: {doc_path}")
                    chunks = self.doc_processor.process_document(doc_path)
                    all_chunks.extend(chunks)
                    processing_stats['documents_processed'] += 1
                    processing_stats['total_chunks'] += len(chunks)
                    
                except Exception as e:
                    error_msg = f"Error processing {doc_path}: {e}"
                    self.logger.error(error_msg)
                    processing_stats['processing_errors'].append(error_msg)
            
            if not all_chunks:
                raise ValueError("No chunks were created from the provided documents")
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            embedded_chunks = self.embedding_generator.embed_documents(all_chunks)
            
            # Add to vector store
            self.logger.info("Adding documents to vector store")
            self.vector_store.add_documents(embedded_chunks)
            
            # Persist vector store
            self.vector_store.persist()
            
            self.is_initialized = True
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            processing_stats.update({
                'end_time': end_time.isoformat(),
                'processing_time_seconds': processing_time,
                'chunks_per_second': len(all_chunks) / processing_time if processing_time > 0 else 0,
                'vector_store_size': self.vector_store.get_collection_size()
            })
            
            self.logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            return processing_stats
            
        except Exception as e:
            self.logger.error(f"Error in document processing: {e}")
            raise
    
    def add_document_content(self, content: str, source_name: str = "uploaded_document") -> Dict[str, Any]:
        """
        Add document content directly without file processing.
        
        Args:
            content: Document content as string
            source_name: Name/identifier for the content
            
        Returns:
            Processing results
        """
        try:
            self.logger.info(f"Processing content from: {source_name}")
            
            # Process content into chunks
            chunks = self.doc_processor.process_text_content(content, source_name)
            
            if not chunks:
                return {'error': 'No chunks created from content', 'chunks_added': 0}
            
            # Generate embeddings
            embedded_chunks = self.embedding_generator.embed_documents(chunks)
            
            # Add to vector store
            self.vector_store.add_documents(embedded_chunks)
            
            # Persist changes
            self.vector_store.persist()
            
            self.is_initialized = True
            
            result = {
                'chunks_added': len(chunks),
                'source_name': source_name,
                'total_collection_size': self.vector_store.get_collection_size()
            }
            
            self.logger.info(f"Added {len(chunks)} chunks from {source_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error adding document content: {e}")
            return {'error': str(e), 'chunks_added': 0}
    
    def query(self, 
              question: str, 
              top_k: int = 5,
              use_reranking: bool = True,
              use_cache: bool = True) -> Dict[str, Any]:
        """
        Process a query and generate an answer using the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of context chunks to retrieve
            use_reranking: Whether to use result reranking
            use_cache: Whether to use query caching
            
        Returns:
            Dictionary with answer, context, and metadata
        """
        if not self.is_initialized:
            return {
                'error': 'Pipeline not initialized. Please add documents first.',
                'answer': '',
                'context': [],
                'metadata': {}
            }
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = create_cache_key(question, top_k) if use_cache and self.cache_dir else None
            
            if cache_key and self.cache_dir:
                cached_result = load_cache(cache_key, self.cache_dir)
                if cached_result:
                    self.logger.info("Returning cached result")
                    return cached_result
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Retrieve relevant context
            if use_reranking:
                context_texts, scores, metadata = self.vector_store.search_with_reranking(
                    query_embedding, question, k=top_k
                )
            else:
                context_texts, scores, metadata = self.vector_store.search(
                    query_embedding, k=top_k
                )
            
            # Generate answer
            answer = self.llm.generate_answer(question, context_texts)
            
            # Format response
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            result = {
                'question': question,
                'answer': answer,
                'context': format_chunks_for_display(context_texts, scores),
                'metadata': {
                    'response_time_seconds': response_time,
                    'chunks_retrieved': len(context_texts),
                    'used_reranking': use_reranking,
                    'from_cache': False,
                    'timestamp': end_time.isoformat()
                }
            }
            
            # Update performance tracking
            self.query_count += 1
            self.total_response_time += response_time
            
            # Cache the result
            if cache_key and self.cache_dir:
                save_cache(cache_key, result, self.cache_dir)
            
            self.logger.info(f"Query processed in {response_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'error': str(e),
                'answer': 'An error occurred while processing your query.',
                'context': [],
                'metadata': {'error': True}
            }
    
    def batch_query(self, questions: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions to process
            **kwargs: Arguments to pass to individual query calls
            
        Returns:
            List of query results
        """
        results = []
        
        for question in questions:
            try:
                result = self.query(question, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in batch query for '{question}': {e}")
                results.append({
                    'error': str(e),
                    'answer': 'Error processing this query.',
                    'context': [],
                    'metadata': {'error': True}
                })
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the pipeline.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            embedding_info = self.embedding_generator.get_model_info()
            llm_info = self.llm.get_model_info()
            
            avg_response_time = (
                self.total_response_time / self.query_count 
                if self.query_count > 0 else 0
            )
            
            return {
                'pipeline_status': 'initialized' if self.is_initialized else 'not_initialized',
                'query_count': self.query_count,
                'average_response_time_seconds': avg_response_time,
                'vector_store': vector_stats,
                'embedding_model': embedding_info,
                'llm_model': llm_info,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline stats: {e}")
            return {'error': str(e)}
    
    def clear_all_data(self) -> None:
        """
        Clear all data from the pipeline (documents and cache).
        """
        try:
            # Clear vector store
            self.vector_store.clear_collection()
            
            # Clear cache if available
            if self.cache_dir and self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.json"):
                    cache_file.unlink()
            
            self.is_initialized = False
            self.query_count = 0
            self.total_response_time = 0.0
            
            self.logger.info("All pipeline data cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing pipeline data: {e}")
            raise
    
    def export_data(self, export_path: Path) -> Dict[str, Any]:
        """
        Export pipeline data for backup or analysis.
        
        Args:
            export_path: Path to save exported data
            
        Returns:
            Export results
        """
        try:
            import json
            
            # Get all data from vector store
            all_results = self.vector_store.collection.get(
                include=['documents', 'metadatas', 'embeddings']
            )
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'pipeline_stats': self.get_pipeline_stats(),
                'documents': all_results.get('documents', []),
                'metadatas': all_results.get('metadatas', []),
                'document_count': len(all_results.get('documents', []))
            }
            
            # Save to file
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Data exported to {export_path}")
            return {'success': True, 'export_path': str(export_path)}
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {'success': False, 'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on all pipeline components.
        
        Returns:
            Health check results
        """
        health_status = {
            'overall_status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check document processor
        try:
            test_chunks = self.doc_processor.chunk_text("This is a test document.")
            health_status['components']['document_processor'] = {
                'status': 'healthy',
                'test_chunks_created': len(test_chunks)
            }
        except Exception as e:
            health_status['components']['document_processor'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check embedding generator
        try:
            test_embedding = self.embedding_generator.generate_single_embedding("test")
            health_status['components']['embedding_generator'] = {
                'status': 'healthy',
                'embedding_dimension': len(test_embedding)
            }
        except Exception as e:
            health_status['components']['embedding_generator'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check vector store
        try:
            collection_size = self.vector_store.get_collection_size()
            health_status['components']['vector_store'] = {
                'status': 'healthy',
                'collection_size': collection_size
            }
        except Exception as e:
            health_status['components']['vector_store'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check LLM
        try:
            test_answer = self.llm.generate_answer("test", ["test context"])
            health_status['components']['llm'] = {
                'status': 'healthy',
                'test_answer_length': len(test_answer)
            }
        except Exception as e:
            health_status['components']['llm'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        return health_status