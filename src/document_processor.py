import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .utils import count_tokens

class DocumentProcessor:
    """
    Document processor with intelligent chunking strategies for RAG systems.
    
    Chunking Strategy Explanation:
    - Token-based chunking (200-500 tokens per chunk)
    - Sentence-aware splitting to maintain semantic coherence
    - Configurable overlap to preserve context between chunks
    - Support for multiple chunking strategies based on document type
    """
    
    def __init__(self, 
                 min_tokens: int = 200,
                 max_tokens: int = 500, 
                 overlap_tokens: int = 50,
                 chunk_strategy: str = "sentence_aware"):
        """
        Initialize the document processor with chunking configuration.
        
        Args:
            min_tokens: Minimum tokens per chunk (prevents very small chunks)
            max_tokens: Maximum tokens per chunk (prevents very large chunks)
            overlap_tokens: Number of overlapping tokens between consecutive chunks
            chunk_strategy: Strategy for chunking ('fixed', 'sentence_aware', 'paragraph_aware')
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.chunk_strategy = chunk_strategy
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DocumentProcessor initialized with strategy: {chunk_strategy}")
        self.logger.info(f"Token range: {min_tokens}-{max_tokens}, overlap: {overlap_tokens}")
    
    def load_document(self, file_path: Path) -> str:
        """
        Load a document from file path with proper encoding handling.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            self.logger.info(f"Loaded document: {file_path} ({len(content)} characters)")
            return content
            
        except UnicodeDecodeError:
            # Try alternative encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    self.logger.warning(f"Used fallback encoding {encoding} for {file_path}")
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise UnicodeDecodeError(f"Could not decode file: {file_path}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content for processing.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
        text = text.strip()
        
        # Remove common artifacts
        text = re.sub(r'\x00', '', text)  # Remove null bytes
        text = re.sub(r'[\x01-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)  # Remove control characters
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Enhanced sentence splitting pattern for medical text
        sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?|\:)\s+'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Handle special cases for medical abbreviations
        processed_sentences = []
        for sentence in sentences:
            # Don't split on common medical abbreviations
            if not re.match(r'^[A-Z]{1,5}\d*$', sentence):  # Skip standalone codes
                processed_sentences.append(sentence)
            elif processed_sentences:  # Merge with previous sentence
                processed_sentences[-1] += ' ' + sentence
            else:
                processed_sentences.append(sentence)
        
        return processed_sentences
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs for paragraph-aware chunking.
        
        Args:
            text: Input text to split
            
        Returns:
            List of paragraphs
        """
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_by_tokens_fixed(self, text: str) -> List[str]:
        """
        Create chunks using fixed token windows.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for word in words:
            word_tokens = count_tokens(word)
            
            if current_tokens + word_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_words = current_chunk[-self.overlap_tokens:] if len(current_chunk) > self.overlap_tokens else current_chunk
                current_chunk = overlap_words + [word]
                current_tokens = count_tokens(' '.join(current_chunk))
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[str]:
        """
        Create chunks based on sentence boundaries while respecting token limits.
        
        This is the recommended chunking strategy as it preserves semantic coherence.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with proper sentence boundaries
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            # If single sentence exceeds max_tokens, split it further
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence into smaller parts
                words = sentence.split()
                for word in words:
                    word_tokens = count_tokens(word)
                    if current_tokens + word_tokens > self.max_tokens and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [word]
                        current_tokens = word_tokens
                    else:
                        current_chunk.append(word)
                        current_tokens += word_tokens
                continue
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                if count_tokens(chunk_text) >= self.min_tokens:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.overlap_tokens > 0 and len(current_chunk) > 1:
                    overlap_sentences = self._get_overlap_sentences(current_chunk, self.overlap_tokens)
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = count_tokens(' '.join(current_chunk))
                else:
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if it meets minimum token requirement
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if count_tokens(chunk_text) >= self.min_tokens:
                chunks.append(chunk_text)
            elif chunks:
                # Merge small final chunk with previous chunk
                chunks[-1] += ' ' + chunk_text
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[str]:
        """
        Create chunks based on paragraph boundaries while respecting token limits.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks based on paragraph structure
        """
        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph)
            
            if current_tokens + paragraph_tokens > self.max_tokens and current_chunk:
                # Save current chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_tokens = paragraph_tokens
            else:
                current_chunk.append(paragraph)
                current_tokens += paragraph_tokens
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """
        Get the last few sentences that fit within the overlap token limit.
        
        Args:
            sentences: List of sentences
            overlap_tokens: Maximum tokens for overlap
            
        Returns:
            List of sentences for overlap
        """
        overlap_sentences = []
        tokens_used = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = count_tokens(sentence)
            if tokens_used + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                tokens_used += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Main chunking method that applies the configured chunking strategy.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Apply the selected chunking strategy
        if self.chunk_strategy == "sentence_aware":
            chunks = self.chunk_by_sentences(text)
        elif self.chunk_strategy == "paragraph_aware":
            chunks = self.chunk_by_paragraphs(text)
        else:  # fixed strategy
            chunks = self.chunk_by_tokens_fixed(text)
        
        # Log chunking results
        self.logger.info(f"Created {len(chunks)} chunks using {self.chunk_strategy} strategy")
        
        if chunks:
            token_counts = [count_tokens(chunk) for chunk in chunks]
            self.logger.info(f"Token distribution - Min: {min(token_counts)}, "
                           f"Max: {max(token_counts)}, Avg: {sum(token_counts)/len(token_counts):.1f}")
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Load and clean the document
        raw_text = self.load_document(file_path)
        cleaned_text = self.clean_text(raw_text)
        
        # Create chunks
        chunks = self.chunk_text(cleaned_text)
        
        # Add metadata to chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{file_path.stem}_chunk_{i:03d}",
                'content': chunk,
                'token_count': count_tokens(chunk),
                'word_count': len(chunk.split()),
                'char_count': len(chunk),
                'source_file': str(file_path),
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            processed_chunks.append(chunk_data)
        
        self.logger.info(f"Processed document into {len(processed_chunks)} chunks")
        return processed_chunks
    
    def process_text_content(self, text: str, source_name: str = "user_input") -> List[Dict[str, Any]]:
        """
        Process raw text content (for uploaded files or direct input).
        
        Args:
            text: Raw text content
            source_name: Name/identifier for the text source
            
        Returns:
            List of chunk dictionaries with metadata
        """
        cleaned_text = self.clean_text(text)
        chunks = self.chunk_text(cleaned_text)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'id': f"{source_name}_chunk_{i:03d}",
                'content': chunk,
                'token_count': count_tokens(chunk),
                'word_count': len(chunk.split()),
                'char_count': len(chunk),
                'source_file': source_name,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            processed_chunks.append(chunk_data)
        
        return processed_chunks