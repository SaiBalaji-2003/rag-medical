import os
import json
import hashlib
import logging
import tiktoken
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

def setup_logging(log_file: Path, level: str = "INFO"):
    """
    Set up logging configuration for the application.
    
    Args:
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    
    Args:
        text: Input text to count tokens for
        model: Model name for tokenizer (default: gpt-3.5-turbo)
    
    Returns:
        Number of tokens in the text
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback to approximate token count
        return len(text.split()) * 1.3

def create_cache_key(query: str, top_k: int = 5) -> str:
    """
    Create a cache key for a query to enable query caching.
    
    Args:
        query: User query string
        top_k: Number of results requested
    
    Returns:
        MD5 hash string to use as cache key
    """
    cache_string = f"{query.lower().strip()}_{top_k}"
    return hashlib.md5(cache_string.encode()).hexdigest()

def save_cache(cache_key: str, data: Dict[str, Any], cache_dir: Path):
    """
    Save query results to cache for faster retrieval.
    
    Args:
        cache_key: Unique identifier for the cached data
        data: Dictionary containing cached data
        cache_dir: Directory to store cache files
    """
    cache_file = cache_dir / f"{cache_key}.json"
    data['timestamp'] = datetime.now().isoformat()
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")

def load_cache(cache_key: str, cache_dir: Path, max_age_hours: int = 24) -> Optional[Dict[str, Any]]:
    """
    Load cached query results if they exist and are not too old.
    
    Args:
        cache_key: Unique identifier for the cached data
        cache_dir: Directory where cache files are stored
        max_age_hours: Maximum age of cache in hours
    
    Returns:
        Cached data if valid, None otherwise
    """
    cache_file = cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check cache age
        cache_time = datetime.fromisoformat(data.get('timestamp', '1970-01-01'))
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
        
        if age_hours > max_age_hours:
            cache_file.unlink()  # Delete old cache
            return None
        
        return data
    except Exception as e:
        logging.warning(f"Failed to load cache: {e}")
        return None

def format_chunks_for_display(chunks: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    """
    Format retrieved chunks for display in the UI.
    
    Args:
        chunks: List of text chunks
        scores: List of similarity scores
    
    Returns:
        List of formatted chunk dictionaries
    """
    formatted_chunks = []
    
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        formatted_chunks.append({
            'rank': i,
            'content': chunk,
            'similarity_score': round(score, 4),
            'word_count': len(chunk.split()),
            'char_count': len(chunk)
        })
    
    return formatted_chunks

def extract_icd_codes(text: str) -> List[str]:
    """
    Extract ICD-10 codes from text using regex patterns.
    
    Args:
        text: Input text to search for ICD codes
    
    Returns:
        List of found ICD-10 codes
    """
    import re
    
    # Pattern for ICD-10 codes (F00-F99 with optional subcodes)
    pattern = r'\b[F][0-9]{2}(?:\.[0-9]{1,2})?\b'
    
    codes = re.findall(pattern, text, re.IGNORECASE)
    return list(set(codes))  # Remove duplicates

def validate_file_upload(file_content: bytes, max_size_mb: int = 10) -> tuple[bool, str]:
    """
    Validate uploaded file content and size.
    
    Args:
        file_content: File content in bytes
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(file_content) == 0:
        return False, "File is empty"
    
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File size ({size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
    
    try:
        # Try to decode as UTF-8
        file_content.decode('utf-8')
        return True, ""
    except UnicodeDecodeError:
        return False, "File must be a valid UTF-8 text file"

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """
    Highlight keywords in text for better readability in UI.
    
    Args:
        text: Input text
        keywords: List of keywords to highlight
    
    Returns:
        Text with highlighted keywords
    """
    import re
    
    highlighted_text = text
    for keyword in keywords:
        if keyword.strip():
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted_text = pattern.sub(f"**{keyword}**", highlighted_text)
    
    return highlighted_text

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and monitoring.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "timestamp": datetime.now().isoformat()
    }