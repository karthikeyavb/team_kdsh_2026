"""
Configuration settings for the consistency checking pipeline.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

@dataclass
class Config:
    # Paths
    novels_dir: str = "data/novels"
    backstories_dir: str = "data/backstories"
    output_dir: str = "outputs"
    vector_store_path: str = "outputs/vector_store"
    
    # Chunking settings
    chunk_size: int = 1024  # tokens
    chunk_overlap: int = 128  # tokens
    min_chunk_size: int = 256  # minimum chunk size
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Retrieval settings
    top_k_chunks: int = 10
    similarity_threshold: float = 0.3
    
    # LLM settings
    llm_model: str = "gpt-4o-mini"  # or "gpt-4", "claude-3-sonnet", etc.
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # OpenAI API
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Aggregation thresholds
    contradiction_threshold: int = 1  # If >= this many contradictions, label 0
    support_ratio_threshold: float = 0.3  # Min ratio of supported claims for label 1
    
    # Processing
    batch_size: int = 10
    max_claims_per_backstory: int = 20
    
    def validate(self):
        """Validate configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        if not os.path.exists(self.novels_dir):
            raise ValueError(f"Novels directory not found: {self.novels_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)

config = Config()