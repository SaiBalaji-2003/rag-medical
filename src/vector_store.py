import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import chromadb

class VectorStore:
    """
    ChromaDB-based vector store with advanced features for RAG systems.
    
    Features:
    - Persistent storage with ChromaDB
    - Efficient similarity search with metadata filtering
    - Support for multiple collections
    - Automatic embedding management
    - Query result ranking and reranking
    """
    
    def __init__(self, 
                 db_path: Path,
                 collection_name: str = "documents",
                 distance_metric: str = "cosine"):
        """
        Initialize the vector store with ChromaDB backend.
        
        Args:
            db_path: Path to store the ChromaDB database
            collection_name: Name of the collection to use
            distance_metric: Distance metric for similarity search
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.distance_metric = distance_metric
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ChromaDB client
        self.client = self._initialize_client()
        self.collection = self._get_or_create_collection()
        
        self.logger.info(f"VectorStore initialized with collection: {collection_name}")
        self.logger.info(f"Database path: {db_path}")
    
    def _initialize_client(self) -> chromadb.Client:
        """
        Initialize ChromaDB client with persistence.
        
        Returns:
            ChromaDB client instance
        """
        try:
            # Ensure database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Configure ChromaDB settings
            client = chromadb.PersistentClient(path=str(self.db_path))
            self.logger.info("ChromaDB client initialized successfully")
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create a new one.
        
        Returns:
            ChromaDB collection instance
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            self.logger.info(f"Retrieved existing collection: {self.collection_name}")
            
        except Exception:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"distance": self.distance_metric}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of document dictionaries with embeddings and metadata
        """
        if not documents:
            self.logger.warning("No documents to add")
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []
            
            for doc in documents:
                # Generate unique ID
                doc_id = doc.get('id', f"doc_{len(ids)}")
                ids.append(doc_id)
                
                # Extract embedding
                embedding = doc.get('embedding')
                if embedding is None:
                    raise ValueError(f"Document {doc_id} missing embedding")
                
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                embeddings.append(embedding)
                
                # Extract content
                content = doc.get('content', '')
                documents_content.append(content)
                
                # Prepare metadata (exclude large fields)
                metadata = {
                    key: value for key, value in doc.items() 
                    if key not in ['embedding', 'content'] and 
                    isinstance(value, (str, int, float, bool))
                }
                metadatas.append(metadata)
            
            # Add to collection in batches to handle large datasets
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_end = min(i + batch_size, len(ids))
                
                self.collection.add(
                    ids=ids[i:batch_end],
                    embeddings=embeddings[i:batch_end],
                    metadatas=metadatas[i:batch_end],
                    documents=documents_content[i:batch_end]
                )
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 5,
               filter_metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Search for similar documents using embedding similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            Tuple of (documents, scores, metadata)
        """
        try:
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.get_collection_size()),
                where=filter_metadata,
                include=['documents', 'distances', 'metadatas']
            )
            
            if not results['documents'] or not results['documents'][0]:
                self.logger.warning("No search results found")
                return [], [], []
            
            # Extract results
            documents = results['documents'][0]
            distances = results['distances'][0]
            metadatas = results['metadatas'][0]
            
            # Convert distances to similarity scores (ChromaDB returns distances)
            similarities = self._distances_to_similarities(distances)
            
            self.logger.info(f"Retrieved {len(documents)} documents from search")
            return documents, similarities, metadatas
            
        except Exception as e:
            self.logger.error(f"Error during vector search: {e}")
            return [], [], []
    
    def _distances_to_similarities(self, distances: List[float]) -> List[float]:
        """
        Convert distance scores to similarity scores.
        
        Args:
            distances: List of distance scores from ChromaDB
            
        Returns:
            List of similarity scores
        """
        if self.distance_metric == "cosine":
            # For cosine distance: similarity = 1 - distance
            return [1.0 - d for d in distances]
        elif self.distance_metric == "l2":
            # For L2 distance: convert to similarity score
            max_dist = max(distances) if distances else 1.0
            return [(max_dist - d) / max_dist for d in distances]
        else:
            # Default: invert distances
            return [1.0 / (1.0 + d) for d in distances]
    
    def search_with_reranking(self,
                             query_embedding: np.ndarray,
                             query_text: str,
                             k: int = 5,
                             rerank_top_k: int = 20) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """
        Search with query-document relevance reranking.
        
        Args:
            query_embedding: Query embedding vector
            query_text: Original query text for reranking
            k: Final number of results to return
            rerank_top_k: Number of candidates to retrieve before reranking
            
        Returns:
            Tuple of (reranked_documents, scores, metadata)
        """
        # Get initial candidates
        documents, scores, metadatas = self.search(
            query_embedding, 
            k=min(rerank_top_k, self.get_collection_size())
        )
        
        if not documents:
            return [], [], []
        
        # Simple keyword-based reranking
        reranked_results = self._rerank_by_keywords(
            query_text, documents, scores, metadatas
        )
        
        # Return top k results
        final_k = min(k, len(reranked_results))
        
        final_docs = [r['document'] for r in reranked_results[:final_k]]
        final_scores = [r['score'] for r in reranked_results[:final_k]]
        final_metadata = [r['metadata'] for r in reranked_results[:final_k]]
        
        return final_docs, final_scores, final_metadata
    
    def _rerank_by_keywords(self, 
                           query: str, 
                           documents: List[str], 
                           scores: List[float], 
                           metadatas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results based on keyword overlap and other heuristics.
        
        Args:
            query: Original query text
            documents: List of document texts
            scores: Original similarity scores
            metadatas: Document metadata
            
        Returns:
            List of reranked results
        """
        import re
        
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        results = []
        
        for doc, score, metadata in zip(documents, scores, metadatas):
            doc_words = set(re.findall(r'\b\w+\b', doc.lower()))
            
            # Calculate keyword overlap
            overlap = len(query_words.intersection(doc_words))
            overlap_ratio = overlap / len(query_words) if query_words else 0
            
            # Boost score based on keyword overlap
            boosted_score = score * (1 + overlap_ratio * 0.2)
            
            # Additional boost for exact phrase matches
            if query.lower() in doc.lower():
                boosted_score *= 1.1
            
            results.append({
                'document': doc,
                'score': boosted_score,
                'metadata': metadata,
                'keyword_overlap': overlap,
                'overlap_ratio': overlap_ratio
            })
        
        # Sort by boosted score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_collection_size(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        try:
            count = self.collection.count()
            return count
        except Exception as e:
            self.logger.error(f"Error getting collection size: {e}")
            return 0
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection and its data.
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            self.logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        try:
            # Get all document IDs
            all_results = self.collection.get(include=['documents'])
            if all_results['ids']:
                self.collection.delete(ids=all_results['ids'])
            
            self.logger.info(f"Cleared all documents from collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.get_collection_size()
            
            # Sample a few documents to get metadata info
            sample_results = self.collection.get(limit=10, include=['metadatas'])
            
            metadata_keys = set()
            for metadata in sample_results.get('metadatas', []):
                if metadata:
                    metadata_keys.update(metadata.keys())
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'metadata_fields': list(metadata_keys),
                'distance_metric': self.distance_metric,
                'database_path': str(self.db_path)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'error': str(e)
            }
    
    def update_document(self, doc_id: str, new_document: Dict[str, Any]) -> None:
        """
        Update an existing document in the collection.
        
        Args:
            doc_id: ID of the document to update
            new_document: Updated document data
        """
        try:
            # Delete existing document
            self.collection.delete(ids=[doc_id])
            
            # Add updated document
            self.add_documents([new_document])
            
            self.logger.info(f"Updated document: {doc_id}")
            
        except Exception as e:
            self.logger.error(f"Error updating document {doc_id}: {e}")
    
    def persist(self) -> None:
        """
        Explicitly persist the collection to disk.
        """
        try:
            self.client.persist()
            self.logger.info("Vector store persisted to disk")
        except Exception as e:
            self.logger.error(f"Error persisting vector store: {e}")