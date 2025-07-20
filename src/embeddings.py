import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    """
    Advanced embedding generator using sentence-transformers with caching and optimization.
    
    Features:
    - Uses all-MiniLM-L6-v2 for high-quality embeddings
    - Automatic GPU/CPU detection and optimization
    - Batch processing for efficient embedding generation
    - Model caching for faster startup
    - Normalized embeddings for better similarity computation
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 cache_folder: Optional[Path] = None,
                 batch_size: int = 32,
                 normalize_embeddings: bool = True):
        """
        Initialize the embedding generator with specified configuration.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            cache_folder: Directory to cache downloaded models
            batch_size: Batch size for processing multiple texts
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.cache_folder = cache_folder
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = self._get_optimal_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Load the model
        self.model = self._load_model()
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        self.logger.info(f"EmbeddingGenerator initialized with {model_name}")
        self.logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for embedding computation.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
            self.logger.info("MPS (Apple Silicon) available")
        else:
            device = "cpu"
            self.logger.info("Using CPU for embeddings")
        
        return device
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the sentence transformer model with error handling.
        
        Returns:
            Loaded SentenceTransformer model
        """
        try:
            model = SentenceTransformer(
                self.model_name, 
                cache_folder=str(self.cache_folder) if self.cache_folder else None,
                device=self.device
            )
            
            # Optimize model for inference
            model.eval()
            if self.device == "cuda":
                model.half()  # Use half precision on GPU for speed
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            
            # Fallback to a smaller model
            fallback_model = "all-MiniLM-L12-v2"
            self.logger.warning(f"Falling back to {fallback_model}")
            
            try:
                model = SentenceTransformer(fallback_model, device=self.device)
                self.model_name = fallback_model
                return model
            except Exception as fallback_error:
                self.logger.error(f"Fallback model also failed: {fallback_error}")
                raise RuntimeError(f"Could not load any embedding model: {fallback_error}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with batch processing.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        try:
            # Remove empty texts and keep track of indices
            valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
            
            if not valid_texts:
                self.logger.warning("All input texts are empty")
                return np.zeros((len(texts), self.embedding_dim))
            
            # Extract valid text content
            valid_indices, valid_text_content = zip(*valid_texts)
            
            # Generate embeddings in batches
            all_embeddings = []
            
            for i in range(0, len(valid_text_content), self.batch_size):
                batch_texts = valid_text_content[i:i + self.batch_size]
                
                # Generate embeddings for this batch
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    batch_size=len(batch_texts),
                    show_progress_bar=False
                )
                
                all_embeddings.append(batch_embeddings)
            
            # Combine all batch embeddings
            if all_embeddings:
                combined_embeddings = np.vstack(all_embeddings)
            else:
                combined_embeddings = np.array([]).reshape(0, self.embedding_dim)
            
            # Create full embedding array with zeros for empty texts
            full_embeddings = np.zeros((len(texts), self.embedding_dim))
            for i, embedding in zip(valid_indices, combined_embeddings):
                full_embeddings[i] = embedding
            
            self.logger.info(f"Generated embeddings for {len(texts)} texts")
            return full_embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return np.zeros((len(texts), self.embedding_dim))
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array embedding with shape (embedding_dim,)
        """
        if not text.strip():
            return np.zeros(self.embedding_dim)
        
        try:
            embedding = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False
            )
            return embedding[0]
            
        except Exception as e:
            self.logger.error(f"Error generating single embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to document chunks.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of document dictionaries with added 'embedding' field
        """
        if not documents:
            return []
        
        # Extract content for embedding
        contents = [doc.get('content', '') for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(contents)
        
        # Add embeddings to documents
        embedded_documents = []
        for doc, embedding in zip(documents, embeddings):
            embedded_doc = doc.copy()
            embedded_doc['embedding'] = embedding
            embedded_doc['embedding_model'] = self.model_name
            embedded_documents.append(embedded_doc)
        
        self.logger.info(f"Added embeddings to {len(embedded_documents)} documents")
        return embedded_documents
    
    def compute_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Array of document embeddings
            
        Returns:
            Array of similarity scores
        """
        if doc_embeddings.size == 0:
            return np.array([])
        
        try:
            # Ensure embeddings are normalized for cosine similarity
            if not self.normalize_embeddings:
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                doc_norms = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
            else:
                query_norm = query_embedding
                doc_norms = doc_embeddings
            
            # Compute cosine similarity
            similarities = np.dot(doc_norms, query_norm)
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return np.zeros(len(doc_embeddings))
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown')
        }
    
    def benchmark_performance(self, sample_texts: List[str]) -> Dict[str, float]:
        """
        Benchmark embedding generation performance.
        
        Args:
            sample_texts: List of sample texts for benchmarking
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        if not sample_texts:
            sample_texts = ["This is a sample text for benchmarking embedding performance."] * 100
        
        # Warm-up run
        self.generate_embeddings(sample_texts[:10])
        
        # Benchmark run
        start_time = time.time()
        embeddings = self.generate_embeddings(sample_texts)
        end_time = time.time()
        
        total_time = end_time - start_time
        texts_per_second = len(sample_texts) / total_time
        
        return {
            "total_time_seconds": total_time,
            "texts_processed": len(sample_texts),
            "texts_per_second": texts_per_second,
            "average_time_per_text_ms": (total_time / len(sample_texts)) * 1000
        }