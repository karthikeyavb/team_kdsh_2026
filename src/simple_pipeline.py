"""
Fallback RAG pipeline for Windows (without Pathway).
Uses simple in-memory vector storage with sentence-transformers.
"""
import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

from .config import config
from .chunking import SmartChunker

class SimpleRAGPipeline:
    """
    A simple in-memory RAG pipeline for environments where Pathway is not available.
    """
    
    def __init__(self):
        self.chunker = SmartChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size
        )
        self.model = None
        self.chunks: Dict[str, List[dict]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.is_initialized = False

    def initialize(self):
        """Initialize by loading model and processing all novels."""
        print("Initializing Simple RAG Pipeline (Windows Fallback)...")
        
        # Load embedding model
        print(f"  Loading embedding model: {config.embedding_model}")
        self.model = SentenceTransformer(config.embedding_model)
        
        # Process novels
        self._process_novels()
        self.is_initialized = True
        print("Simple RAG Pipeline initialized.")

    def _process_novels(self):
        """Read and chunk all novels."""
        for filename in os.listdir(config.novels_dir):
            if not filename.endswith('.txt'):
                continue
                
            story_id = os.path.splitext(filename)[0]
            filepath = os.path.join(config.novels_dir, filename)
            
            print(f"  Processing novel: {story_id}")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Create chunks
            chunks = self.chunker.chunk_text(content, story_id)
            self.chunks[story_id] = chunks
            print(f"    Created {len(chunks)} chunks")
            
            # Embed chunks
            self._embed_story_chunks(story_id, chunks)

    def _embed_story_chunks(self, story_id: str, chunks: List[dict]):
        """Embed chunks for a story."""
        texts = [c['content'] for c in chunks]
        
        if not texts:
            return

        print(f"    Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False)
        self.embeddings[story_id] = embeddings

    def retrieve_for_claim(
        self, 
        claim: str, 
        story_id: str, 
        top_k: int = 10
    ) -> List[dict]:
        """Retrieve relevant chunks for a claim."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized")
            
        if story_id not in self.embeddings:
            return []

        # Embed claim
        claim_emb = self.model.encode([claim])[0]
        
        # Calculate scores (cosine similarity)
        story_embs = self.embeddings[story_id]
        
        # Normalize
        norm_claim = np.linalg.norm(claim_emb)
        norm_story = np.linalg.norm(story_embs, axis=1)
        
        scores = np.dot(story_embs, claim_emb) / (norm_story * norm_claim + 1e-8)
        
        # Get top k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= config.similarity_threshold:
                chunk = self.chunks[story_id][idx]
                results.append({
                    'chunk': chunk,
                    'score': score,
                    'content': chunk['content'],
                    'chunk_id': chunk['chunk_id']
                })
                
        return results

    def get_story_chunks(self, story_id: str) -> List[dict]:
        """Get all chunks for a story."""
        return self.chunks.get(story_id, [])
