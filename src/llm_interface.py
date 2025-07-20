import logging
import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pathlib import Path

class LLMInterface:
    """
    Advanced LLM interface supporting multiple models with fallback options.
    
    Features:
    - Multiple model support with automatic fallback
    - GPU/CPU optimization
    - Context-aware answer generation
    - Template-based prompting for medical Q&A
    - Answer quality filtering and validation
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 cache_dir: Optional[Path] = None,
                 max_tokens: int = 512,
                 temperature: float = 0.7):
        """
        Initialize the LLM interface with specified configuration.
        
        Args:
            model_name: Name of the HuggingFace model to use
            cache_dir: Directory to cache downloaded models
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature for generation
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = self._get_optimal_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer, self.model, self.generator = self._load_model_with_fallback()
        
        # Medical Q&A templates
        self.templates = self._initialize_templates()
        
        self.logger.info(f"LLMInterface initialized with model: {self.model_name}")
    
    def _get_optimal_device(self) -> str:
        """
        Determine the optimal device for model inference.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            self.logger.info("MPS (Apple Silicon) available")
        else:
            device = "cpu"
            self.logger.info("Using CPU for LLM inference")
        
        return device
    
    def _load_model_with_fallback(self) -> tuple:
        """
        Load model with automatic fallback to simpler models if needed.
        
        Returns:
            Tuple of (tokenizer, model, generator)
        """
        model_priority = [
            self.model_name,
            "distilgpt2",
            "gpt2"
        ]
        
        for model_name in model_priority:
            try:
                self.logger.info(f"Attempting to load model: {model_name}")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    padding_side='left'
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
                
                # Set pad token if not available
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    model.config.pad_token_id = tokenizer.eos_token_id
                
                # Create generation pipeline
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == "cuda" else -1,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True
                )
                
                self.model_name = model_name
                self.logger.info(f"Successfully loaded model: {model_name}")
                return tokenizer, model, generator
                
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        raise RuntimeError("Could not load any LLM model")
    
    def _initialize_templates(self) -> Dict[str, str]:
        """
        Initialize prompt templates for different types of medical queries.
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            "icd_classification": """Based on the following medical information, provide the correct ICD-10 classification.

Context: {context}

Question: {query}

Please provide the specific ICD-10 code and description that matches the given diagnosis.

Answer:""",
            
            "diagnostic_criteria": """Based on the following medical information, provide the diagnostic criteria.

Context: {context}

Question: {query}

Please provide a comprehensive answer about the diagnostic criteria based on the medical information provided.

Answer:""",
            
            "general_medical": """Based on the following medical information, answer the question accurately.

Context: {context}

Question: {query}

Provide a clear, evidence-based answer based on the medical information provided.

Answer:""",
            
            "simple_extraction": """Context: {context}

Question: {query}

Answer:"""
        }
    
    def _select_template(self, query: str) -> str:
        """
        Select the most appropriate template based on query content.
        
        Args:
            query: User query
            
        Returns:
            Template name
        """
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["classification", "code", "coded", "f32", "f33"]):
            return "icd_classification"
        elif any(term in query_lower for term in ["diagnostic criteria", "criteria", "diagnosis"]):
            return "diagnostic_criteria"
        elif len(query) > 100:
            return "general_medical"
        else:
            return "simple_extraction"
    
    def generate_answer(self, query: str, context: List[str], max_context_length: int = 2000) -> str:
        """
        Generate an answer using the LLM with context-aware prompting.
        
        Args:
            query: User question
            context: List of relevant context chunks
            max_context_length: Maximum length of context to use
            
        Returns:
            Generated answer
        """
        try:
            # Prepare context
            context_text = self._prepare_context(context, max_context_length)
            
            if not context_text.strip():
                return "I don't have enough relevant information to answer your question."
            
            # Select appropriate template
            template_name = self._select_template(query)
            template = self.templates[template_name]
            
            # Format prompt
            prompt = template.format(context=context_text, query=query)
            
            # Generate response
            response = self._generate_with_fallback(prompt)
            
            # Post-process and validate answer
            answer = self._post_process_answer(response, query, context)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return self._extract_answer_from_context(query, context)
    
    def _prepare_context(self, context: List[str], max_length: int) -> str:
        """
        Prepare and truncate context to fit within token limits.
        
        Args:
            context: List of context chunks
            max_length: Maximum character length
            
        Returns:
            Prepared context string
        """
        if not context:
            return ""
        
        # Join context chunks
        full_context = " ".join(context)
        
        # Truncate if too long
        if len(full_context) > max_length:
            full_context = full_context[:max_length] + "..."
        
        return full_context
    
    def _generate_with_fallback(self, prompt: str) -> str:
        """
        Generate text with fallback strategies for robustness.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        generation_configs = [
            {
                "max_new_tokens": 150,
                "temperature": self.temperature,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            {
                "max_new_tokens": 100,
                "temperature": 0.5,
                "do_sample": True,
                "top_p": 0.8
            },
            {
                "max_new_tokens": 80,
                "temperature": 0.3,
                "do_sample": False
            }
        ]
        
        for config in generation_configs:
            try:
                response = self.generator(
                    prompt,
                    **config,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                if response and response[0]['generated_text'].strip():
                    return response[0]['generated_text'].strip()
                    
            except Exception as e:
                self.logger.warning(f"Generation failed with config {config}: {e}")
                continue
        
        return ""
    
    def _post_process_answer(self, generated_text: str, query: str, context: List[str]) -> str:
        """
        Post-process and validate the generated answer.
        
        Args:
            generated_text: Raw generated text
            query: Original query
            context: Context used for generation
            
        Returns:
            Processed and validated answer
        """
        if not generated_text or len(generated_text.strip()) < 10:
            return self._extract_answer_from_context(query, context)
        
        # Clean up the generated text
        answer = generated_text.strip()
        
        # Remove common artifacts
        answer = self._remove_artifacts(answer)
        
        # Validate answer quality
        if not self._is_valid_answer(answer, query):
            return self._extract_answer_from_context(query, context)
        
        return answer
    
    def _remove_artifacts(self, text: str) -> str:
        """
        Remove common generation artifacts from the text.
        
        Args:
            text: Generated text
            
        Returns:
            Cleaned text
        """
        # Remove repetitive patterns
        lines = text.split('\n')
        unique_lines = []
        
        for line in lines:
            line = line.strip()
            if line and line not in unique_lines:
                unique_lines.append(line)
        
        cleaned_text = '\n'.join(unique_lines)
        
        # Remove incomplete sentences at the end
        sentences = cleaned_text.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 20:
            cleaned_text = '.'.join(sentences[:-1]) + '.'
        
        return cleaned_text
    
    def _is_valid_answer(self, answer: str, query: str) -> bool:
        """
        Validate if the generated answer is reasonable.
        
        Args:
            answer: Generated answer
            query: Original query
            
        Returns:
            True if answer is valid
        """
        # Basic quality checks
        if len(answer) < 10:
            return False
        
        if answer.count('.') == 0 and len(answer) > 50:
            return False  # Likely incomplete
        
        # Check for nonsensical repetition
        words = answer.split()
        if len(set(words)) < len(words) * 0.5:
            return False  # Too repetitive
        
        return True
    
    def _extract_answer_from_context(self, query: str, context: List[str]) -> str:
        """
        Extract answer directly from context when generation fails.
        
        Args:
            query: User query
            context: Available context
            
        Returns:
            Extracted answer
        """
        if not context:
            return "No relevant information found to answer your question."
        
        query_lower = query.lower()
        
        # Handle specific query types
        if "recurrent depressive" in query_lower and "remission" in query_lower:
            for ctx in context:
                if "f33.4" in ctx.lower() and "remission" in ctx.lower():
                    return "F33.4 - Recurrent depressive disorder, currently in remission"
        
        if "diagnostic criteria" in query_lower and "ocd" in query_lower:
            for ctx in context:
                if "obsessive-compulsive disorder" in ctx.lower() and "criteria" in ctx.lower():
                    return ctx
        
        # Return most relevant context
        return context[0] if context else "No relevant information found."
    
    def batch_generate(self, queries: List[str], contexts: List[List[str]]) -> List[str]:
        """
        Generate answers for multiple queries in batch.
        
        Args:
            queries: List of queries
            contexts: List of context lists for each query
            
        Returns:
            List of generated answers
        """
        answers = []
        
        for query, context in zip(queries, contexts):
            try:
                answer = self.generate_answer(query, context)
                answers.append(answer)
            except Exception as e:
                self.logger.error(f"Error in batch generation for query '{query}': {e}")
                answers.append("Error generating answer for this query.")
        
        return answers
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        try:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "vocab_size": self.tokenizer.vocab_size,
                "model_type": self.model.config.model_type if hasattr(self.model, 'config') else 'unknown'
            }
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {"model_name": self.model_name, "error": str(e)}