"""
Smart chunking module for splitting novels into meaningful segments.
Handles chapter detection, paragraph preservation, and overlap.
"""
import re
from typing import List, Dict, Tuple, Optional
import hashlib


class SmartChunker:
    """
    Intelligent text chunker that respects document structure.
    """
    
    # Common chapter patterns
    CHAPTER_PATTERNS = [
        r'^chapter\s+\d+',
        r'^chapter\s+[ivxlcdm]+',
        r'^part\s+\d+',
        r'^part\s+[ivxlcdm]+',
        r'^book\s+\d+',
        r'^\d+\.',
        r'^[ivxlcdm]+\.',
        r'^prologue',
        r'^epilogue',
        r'^introduction',
        r'^preface',
    ]
    
    def __init__(
        self, 
        chunk_size: int = 1024,
        overlap: int = 128,
        min_chunk_size: int = 256
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.chapter_regex = re.compile(
            '|'.join(self.CHAPTER_PATTERNS), 
            re.IGNORECASE | re.MULTILINE
        )
    
    def chunk_text(self, text: str, story_id: str) -> List[Dict]:
        """
        Split text into chunks with metadata.
        
        Returns list of chunk dictionaries with:
        - chunk_id: unique identifier
        - story_id: parent story identifier
        - chunk_index: position in sequence
        - content: chunk text
        - chapter_info: detected chapter if any
        - start_pos: character start position
        - end_pos: character end position
        """
        # First, try to detect chapters
        chapters = self._detect_chapters(text)
        
        if chapters:
            # Chunk within chapters
            chunks = self._chunk_by_chapters(text, chapters, story_id)
        else:
            # Fall back to paragraph-aware chunking
            chunks = self._chunk_by_paragraphs(text, story_id)
        
        return chunks
    
    def _detect_chapters(self, text: str) -> List[Tuple[int, str]]:
        """
        Detect chapter boundaries in the text.
        Returns list of (position, chapter_title) tuples.
        """
        chapters = []
        
        for match in self.chapter_regex.finditer(text):
            # Get the full line as chapter title
            line_start = text.rfind('\n', 0, match.start()) + 1
            line_end = text.find('\n', match.end())
            if line_end == -1:
                line_end = len(text)
            
            chapter_title = text[line_start:line_end].strip()
            chapters.append((match.start(), chapter_title))
        
        return chapters
    
    def _chunk_by_chapters(
        self, 
        text: str, 
        chapters: List[Tuple[int, str]], 
        story_id: str
    ) -> List[Dict]:
        """Chunk text respecting chapter boundaries."""
        chunks = []
        chunk_index = 0
        
        for i, (start_pos, chapter_title) in enumerate(chapters):
            # Determine chapter end
            if i + 1 < len(chapters):
                end_pos = chapters[i + 1][0]
            else:
                end_pos = len(text)
            
            chapter_text = text[start_pos:end_pos]
            
            # Chunk within chapter
            chapter_chunks = self._chunk_text_segment(
                chapter_text,
                story_id,
                chunk_index,
                chapter_title,
                start_pos
            )
            
            chunks.extend(chapter_chunks)
            chunk_index += len(chapter_chunks)
        
        # Handle text before first chapter
        if chapters and chapters[0][0] > 0:
            preamble = text[:chapters[0][0]]
            if len(preamble.strip()) > self.min_chunk_size:
                preamble_chunks = self._chunk_text_segment(
                    preamble, story_id, 0, "Preamble", 0
                )
                # Renumber all chunks
                for chunk in chunks:
                    chunk['chunk_index'] += len(preamble_chunks)
                chunks = preamble_chunks + chunks
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, story_id: str) -> List[Dict]:
        """Chunk text by paragraphs when no chapters detected."""
        return self._chunk_text_segment(text, story_id, 0, "Unknown", 0)
    
    def _chunk_text_segment(
        self,
        text: str,
        story_id: str,
        start_index: int,
        chapter_info: str,
        base_position: int
    ) -> List[Dict]:
        """
        Chunk a text segment respecting paragraph boundaries.
        """
        chunks = []
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = []
        current_length = 0
        current_start = 0
        chunk_index = start_index
        
        for para_start, para_text in paragraphs:
            para_length = len(para_text.split())
            
            if current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = '\n\n'.join(current_chunk)
                if len(chunk_content.strip()) >= self.min_chunk_size:
                    chunks.append(self._create_chunk_dict(
                        content=chunk_content,
                        story_id=story_id,
                        chunk_index=chunk_index,
                        chapter_info=chapter_info,
                        start_pos=base_position + current_start,
                        end_pos=base_position + para_start
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text.split()) if overlap_text else 0
                current_start = para_start
            
            current_chunk.append(para_text)
            current_length += para_length
        
        # Save final chunk
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            if len(chunk_content.strip()) >= self.min_chunk_size:
                chunks.append(self._create_chunk_dict(
                    content=chunk_content,
                    story_id=story_id,
                    chunk_index=chunk_index,
                    chapter_info=chapter_info,
                    start_pos=base_position + current_start,
                    end_pos=base_position + len(text)
                ))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[Tuple[int, str]]:
        """Split text into paragraphs with their positions."""
        paragraphs = []
        current_pos = 0
        
        # Split on double newlines or multiple newlines
        parts = re.split(r'\n\s*\n', text)
        
        for part in parts:
            part = part.strip()
            if part:
                # Find actual position
                pos = text.find(part, current_pos)
                if pos != -1:
                    paragraphs.append((pos, part))
                    current_pos = pos + len(part)
        
        return paragraphs
    
    def _get_overlap_text(self, chunks: List[str]) -> str:
        """Get overlap text from end of current chunks."""
        if not chunks:
            return ""
        
        # Take words from the last chunk(s) for overlap
        all_text = ' '.join(chunks)
        words = all_text.split()
        
        overlap_words = words[-self.overlap:] if len(words) > self.overlap else words
        return ' '.join(overlap_words)
    
    def _create_chunk_dict(
        self,
        content: str,
        story_id: str,
        chunk_index: int,
        chapter_info: str,
        start_pos: int,
        end_pos: int
    ) -> Dict:
        """Create a chunk dictionary with all metadata."""
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{story_id}_{chunk_index}_{content[:100]}".encode()
        ).hexdigest()[:12]
        
        return {
            'chunk_id': chunk_id,
            'story_id': story_id,
            'chunk_index': chunk_index,
            'content': content.strip(),
            'chapter_info': chapter_info,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'word_count': len(content.split())
        }