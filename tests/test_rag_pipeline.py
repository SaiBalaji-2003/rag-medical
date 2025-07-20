import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_interface import LLMInterface
from src.rag_pipeline import RAGPipeline
from src.utils import count_tokens, create_cache_key

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""
    
    def setUp(self):
        self.processor = DocumentProcessor()
        self.sample_text = """
        F33.4 Recurrent depressive disorder, currently in remission: There has been at least one previous mild, moderate, or severe depressive episode, but the current mental state does not meet criteria for depressive episode of any severity.
        
        F42 - Obsessive-compulsive disorder: A disorder characterized by recurrent obsessional thoughts or compulsive acts. Obsessional thoughts are ideas, images, or impulses that enter the patient's mind again and again in a stereotyped form.
        """
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "This   is\n\n\na  test\n  text."
        cleaned = self.processor.clean_text(dirty_text)
        self.assertNotIn('\n\n\n', cleaned)
        self.assertNotIn('  ', cleaned)
    
    def test_chunk_text(self):
        """Test text chunking functionality."""
        chunks = self.processor.chunk_text(self.sample_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            token_count = count_tokens(chunk)
            self.assertGreaterEqual(token_count, self.processor.min_tokens)
            self.assertLessEqual(token_count, self.processor.max_tokens)
    
    def test_sentence_splitting(self):
        """Test sentence splitting functionality."""
        sentences = self.processor.split_into_sentences(self.sample_text)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
    
    def test_process_text_content(self):
        """Test processing text content."""
        chunks = self.processor.process_text_content(self.sample_text, "test_source")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertIn('id', chunk)
            self.assertIn('content', chunk)
            self.assertIn('token_count', chunk)
            self.assertIn('source_file', chunk)

class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for EmbeddingGenerator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class with embedding generator."""
        cls.embedding_generator = EmbeddingGenerator()
    
    def test_single_embedding(self):
        """Test single text embedding generation."""
        text = "This is a test sentence for embedding."
        embedding = self.embedding_generator.generate_single_embedding(text)
        
        self.assertEqual(len(embedding), self.embedding_generator.embedding_dim)
        self.assertTrue(all(isinstance(x, (int, float)) for x in embedding))
    
    def test_batch_embeddings(self):
        """Test batch embedding generation."""
        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        self.assertEqual(embeddings.shape[0], len(texts))
        self.assertEqual(embeddings.shape[1], self.embedding_generator.embedding_dim)
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        empty_embedding = self.embedding_generator.generate_single_embedding("")
        self.assertEqual(len(empty_embedding), self.embedding_generator.embedding_dim)
        
        mixed_texts = ["Valid text", "", "Another valid text"]
        embeddings = self.embedding_generator.generate_embeddings(mixed_texts)
        self.assertEqual(embeddings.shape[0], len(mixed_texts))
    
    def test_embed_documents(self):
        """Test embedding documents with metadata."""
        documents = [
            {'content': 'First document content', 'id': 'doc1'},
            {'content': 'Second document content', 'id': 'doc2'}
        ]
        
        embedded_docs = self.embedding_generator.embed_documents(documents)
        
        self.assertEqual(len(embedded_docs), len(documents))
        for doc in embedded_docs:
            self.assertIn('embedding', doc)
            self.assertIn('embedding_model', doc)

class TestVectorStore(unittest.TestCase):
    """Test cases for VectorStore class."""
    
    def setUp(self):
        """Set up test with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_vectordb"
        self.vector_store = VectorStore(self.db_path, "test_collection")
        
        # Create test documents with embeddings
        self.test_documents = [
            {
                'id': 'doc1',
                'content': 'Depression is a mental health disorder.',
                'embedding': [0.1, 0.2, 0.3, 0.4],
                'source': 'test'
            },
            {
                'id': 'doc2', 
                'content': 'Anxiety disorders affect many people.',
                'embedding': [0.5, 0.6, 0.7, 0.8],
                'source': 'test'
            }
        ]
    
    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_documents(self):
        """Test adding documents to vector store."""
        self.vector_store.add_documents(self.test_documents)
        collection_size = self.vector_store.get_collection_size()
        self.assertEqual(collection_size, len(self.test_documents))
    
    def test_search(self):
        """Test vector similarity search."""
        self.vector_store.add_documents(self.test_documents)
        
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        documents, scores, metadata = self.vector_store.search(query_embedding, k=2)
        
        self.assertIsInstance(documents, list)
        self.assertIsInstance(scores, list)
        self.assertIsInstance(metadata, list)
        self.assertLessEqual(len(documents), 2)
    
    def test_collection_stats(self):
        """Test getting collection statistics."""
        self.vector_store.add_documents(self.test_documents)
        stats = self.vector_store.get_collection_stats()
        
        self.assertIn('collection_name', stats)
        self.assertIn('document_count', stats)
        self.assertEqual(stats['document_count'], len(self.test_documents))

class TestLLMInterface(unittest.TestCase):
    """Test cases for LLMInterface class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up LLM interface."""
        cls.llm = LLMInterface()
    
    def test_generate_answer(self):
        """Test answer generation."""
        query = "What is depression?"
        context = ["Depression is a mental health disorder characterized by persistent sadness."]
        
        answer = self.llm.generate_answer(query, context)
        
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
    
    def test_context_extraction(self):
        """Test fallback context extraction."""
        query = "What is F33.4?"
        context = ["F33.4 Recurrent depressive disorder, currently in remission"]
        
        answer = self.llm._extract_answer_from_context(query, context)
        self.assertIsInstance(answer, str)
        self.assertIn("F33.4", answer)
    
    def test_batch_generation(self):
        """Test batch answer generation."""
        queries = ["What is depression?", "What is anxiety?"]
        contexts = [
            ["Depression is a mood disorder."],
            ["Anxiety is an emotional response."]
        ]
        
        answers = self.llm.batch_generate(queries, contexts)
        
        self.assertEqual(len(answers), len(queries))
        for answer in answers:
            self.assertIsInstance(answer, str)

class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAGPipeline integration."""
    
    def setUp(self):
        """Set up test with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.vectordb_path = Path(self.temp_dir) / "vectordb"
        self.cache_dir = Path(self.temp_dir) / "cache"
        self.models_dir = Path(self.temp_dir) / "models"
        
        self.cache_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.pipeline = RAGPipeline(
            vectordb_path=self.vectordb_path,
            cache_dir=self.cache_dir,
            models_dir=self.models_dir
        )
    
    def tearDown(self):
        """Clean up test directories."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_initialization(self):
        """Test pipeline component initialization."""
        self.assertIsNotNone(self.pipeline.doc_processor)
        self.assertIsNotNone(self.pipeline.embedding_generator)
        self.assertIsNotNone(self.pipeline.vector_store)
        self.assertIsNotNone(self.pipeline.llm)
    
    def test_add_document_content(self):
        """Test adding document content to pipeline."""
        content = """
        F33.4 Recurrent depressive disorder, currently in remission: There has been at least one previous episode, but current state does not meet criteria for depression.
        
        F42 Obsessive-compulsive disorder: Characterized by recurrent obsessions or compulsions.
        """
        
        result = self.pipeline.add_document_content(content, "test_doc")
        
        self.assertNotIn('error', result)
        self.assertGreater(result['chunks_added'], 0)
        self.assertTrue(self.pipeline.is_initialized)
    
    def test_query_processing(self):
        """Test query processing pipeline."""
        # Add some content first
        content = "F33.4 Recurrent depressive disorder, currently in remission."
        self.pipeline.add_document_content(content, "test_doc")
        
        # Test query
        query = "What is F33.4?"
        result = self.pipeline.query(query, top_k=2)
        
        self.assertNotIn('error', result)
        self.assertIn('answer', result)
        self.assertIn('context', result)
        self.assertIn('metadata', result)
    
    def test_health_check(self):
        """Test pipeline health check."""
        health_status = self.pipeline.health_check()
        
        self.assertIn('overall_status', health_status)
        self.assertIn('components', health_status)
        self.assertIn('timestamp', health_status)
    
    def test_pipeline_stats(self):
        """Test getting pipeline statistics."""
        stats = self.pipeline.get_pipeline_stats()
        
        self.assertIn('pipeline_status', stats)
        self.assertIn('query_count', stats)
        self.assertIn('vector_store', stats)
        self.assertIn('embedding_model', stats)
        self.assertIn('llm_model', stats)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_count_tokens(self):
        """Test token counting functionality."""
        text = "This is a test sentence."
        token_count = count_tokens(text)
        
        self.assertIsInstance(token_count, (int, float))
        self.assertGreater(token_count, 0)
    
    def test_create_cache_key(self):
        """Test cache key creation."""
        query = "What is depression?"
        cache_key = create_cache_key(query, top_k=5)
        
        self.assertIsInstance(cache_key, str)
        self.assertEqual(len(cache_key), 32)  # MD5 hash length
        
        # Same input should produce same key
        cache_key2 = create_cache_key(query, top_k=5)
        self.assertEqual(cache_key, cache_key2)
    
    def test_file_validation(self):
        """Test file upload validation."""
        from src.utils import validate_file_upload
        
        # Valid text content
        valid_content = b"This is valid text content."
        is_valid, error = validate_file_upload(valid_content)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Empty content
        empty_content = b""
        is_valid, error = validate_file_upload(empty_content)
        self.assertFalse(is_valid)
        self.assertIn("empty", error.lower())

if __name__ == '__main__':
    # Create test suite
    test_classes = [
        TestDocumentProcessor,
        TestEmbeddingGenerator,
        TestVectorStore,
        TestLLMInterface,
        TestRAGPipeline,
        TestUtils
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with error code if tests failed
    if not result.wasSuccessful():
        sys.exit(1)