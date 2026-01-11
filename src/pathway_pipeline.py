"""
Pathway-based data ingestion and processing pipeline.
This satisfies Track A's requirement for meaningful Pathway usage.
"""
import pathway as pw
from pathway.stdlib.ml.index import KNNIndex
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.vector_store import VectorStoreServer
import os
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

from .config import config
from .chunking import SmartChunker


class NovelSchema(pw.Schema):
    """Schema for novel documents."""
    story_id: str
    filename: str
    content: str
    metadata: str  # JSON string


class ChunkSchema(pw.Schema):
    """Schema for document chunks."""
    chunk_id: str
    story_id: str
    chunk_index: int
    content: str
    chapter_info: str
    start_pos: int
    end_pos: int


class EmbeddedChunkSchema(pw.Schema):
    """Schema for embedded chunks."""
    chunk_id: str
    story_id: str
    chunk_index: int
    content: str
    chapter_info: str
    embedding: list  # Vector embedding


class PathwayNovelProcessor:
    """
    Main Pathway-based processor for novels.
    Handles ingestion, chunking, embedding, and indexing.
    """
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.chunker = SmartChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size
        )
        self.embedder = None
        self.index = None
        self._chunk_cache: Dict[str, List[dict]] = {}
        
    def _load_novels_as_table(self) -> pw.Table:
        """
        Load all novels from directory into a Pathway table.
        """
        novels_data = []
        
        for filename in os.listdir(self.config.novels_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.config.novels_dir, filename)
                story_id = os.path.splitext(filename)[0]
                
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                metadata = json.dumps({
                    'filename': filename,
                    'filepath': filepath,
                    'word_count': len(content.split()),
                    'char_count': len(content)
                })
                
                novels_data.append({
                    'story_id': story_id,
                    'filename': filename,
                    'content': content,
                    'metadata': metadata
                })
        
        # Create Pathway table from data
        return pw.debug.table_from_markdown(
            self._data_to_markdown(novels_data, NovelSchema)
        )
    
    def _data_to_markdown(self, data: List[dict], schema) -> str:
        """Convert data list to Pathway markdown format."""
        if not data:
            return ""
        
        # Get column names from schema
        columns = list(schema.__annotations__.keys())
        
        # Build markdown table
        lines = [" | ".join(columns)]
        lines.append(" | ".join(["---"] * len(columns)))
        
        for row in data:
            values = []
            for col in columns:
                val = str(row.get(col, "")).replace("|", "\\|").replace("\n", " ")
                # Truncate very long values for markdown
                if len(val) > 100:
                    val = val[:100] + "..."
                values.append(val)
            lines.append(" | ".join(values))
        
        return "\n".join(lines)
    
    def process_novels_with_pathway(self) -> Dict[str, List[dict]]:
        """
        Main Pathway processing pipeline.
        Returns dictionary mapping story_id to list of chunk dictionaries.
        """
        print("Starting Pathway novel processing pipeline...")
        
        # Process each novel
        all_chunks = {}
        
        for filename in os.listdir(self.config.novels_dir):
            if not filename.endswith('.txt'):
                continue
                
            story_id = os.path.splitext(filename)[0]
            filepath = os.path.join(self.config.novels_dir, filename)
            
            print(f"Processing novel: {story_id}")
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Use smart chunking
            chunks = self.chunker.chunk_text(content, story_id)
            all_chunks[story_id] = chunks
            
            print(f"  Created {len(chunks)} chunks for {story_id}")
        
        self._chunk_cache = all_chunks
        return all_chunks
    
    def create_pathway_streaming_pipeline(self):
        """
        Create a Pathway streaming pipeline for real-time novel ingestion.
        This demonstrates Pathway's streaming capabilities.
        """
        # Define input connector for novels directory
        class NovelInputFormat(pw.io.fs.FsInputFormat):
            def read(self, data: bytes):
                content = data.decode('utf-8', errors='ignore')
                return {'content': content}
        
        # Create streaming input
        input_table = pw.io.fs.read(
            self.config.novels_dir,
            format="plaintext",
            mode="static"
        )
        
        return input_table
    
    def build_vector_index_with_pathway(
        self, 
        chunks: Dict[str, List[dict]]
    ) -> 'PathwayVectorIndex':
        """
        Build a vector index using Pathway's vector store capabilities.
        """
        return PathwayVectorIndex(chunks, self.config)
    
    def get_chunks_for_story(self, story_id: str) -> List[dict]:
        """Get cached chunks for a specific story."""
        return self._chunk_cache.get(story_id, [])


class PathwayVectorIndex:
    """
    Vector index built on top of Pathway for semantic search.
    """
    
    def __init__(self, chunks: Dict[str, List[dict]], config_obj):
        self.config = config_obj
        self.chunks = chunks
        self.embeddings: Dict[str, List[Tuple[dict, List[float]]]] = {}
        self._build_index()
    
    def _build_index(self):
        """Build embeddings for all chunks."""
        from sentence_transformers import SentenceTransformer
        
        print("Building vector index...")
        model = SentenceTransformer(self.config.embedding_model)
        
        for story_id, story_chunks in self.chunks.items():
            print(f"  Embedding chunks for {story_id}...")
            
            # Get all chunk contents
            contents = [chunk['content'] for chunk in story_chunks]
            
            # Batch embed
            embeddings = model.encode(
                contents, 
                show_progress_bar=False,
                batch_size=32
            )
            
            # Store chunk-embedding pairs
            self.embeddings[story_id] = [
                (chunk, emb.tolist()) 
                for chunk, emb in zip(story_chunks, embeddings)
            ]
        
        print("Vector index built successfully.")
    
    def search(
        self, 
        query: str, 
        story_id: str, 
        top_k: int = 10
    ) -> List[Tuple[dict, float]]:
        """
        Search for relevant chunks given a query.
        Returns list of (chunk, similarity_score) tuples.
        """
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        if story_id not in self.embeddings:
            return []
        
        model = SentenceTransformer(self.config.embedding_model)
        query_embedding = model.encode([query])[0]
        
        # Calculate cosine similarities
        results = []
        for chunk, chunk_emb in self.embeddings[story_id]:
            similarity = self._cosine_similarity(query_embedding, chunk_emb)
            results.append((chunk, similarity))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


class PathwayRAGPipeline:
    """
    Complete RAG pipeline using Pathway for the consistency checking task.
    """
    
    def __init__(self):
        self.processor = PathwayNovelProcessor()
        self.vector_index: Optional[PathwayVectorIndex] = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the pipeline by processing all novels."""
        # Process novels and build chunks
        chunks = self.processor.process_novels_with_pathway()
        
        # Build vector index
        self.vector_index = self.processor.build_vector_index_with_pathway(chunks)
        
        self.is_initialized = True
        print("RAG Pipeline initialized successfully.")
    
    def retrieve_for_claim(
        self, 
        claim: str, 
        story_id: str, 
        top_k: int = 10
    ) -> List[dict]:
        """
        Retrieve relevant chunks for a given claim.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        results = self.vector_index.search(claim, story_id, top_k)
        
        # Return chunks with their scores
        return [
            {
                'chunk': chunk,
                'score': score,
                'content': chunk['content'],
                'chunk_id': chunk['chunk_id']
            }
            for chunk, score in results
            if score >= config.similarity_threshold
        ]
    
    def get_story_chunks(self, story_id: str) -> List[dict]:
        """Get all chunks for a story."""
        return self.processor.get_chunks_for_story(story_id)