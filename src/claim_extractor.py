"""
Extract testable claims from character backstories.
"""
import re
from typing import List, Dict, Optional
import json
from openai import OpenAI

from .config import config


class ClaimExtractor:
    """
    Extracts individual testable claims from backstory text.
    """
    
    CLAIM_CATEGORIES = [
        'early_life',
        'family',
        'education',
        'formative_events',
        'beliefs',
        'fears',
        'ambitions',
        'relationships',
        'personality_traits',
        'physical_attributes',
        'skills_abilities',
        'occupation',
        'location_history'
    ]
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            self.client = OpenAI(api_key=config.openai_api_key)
    
    def extract_claims(
        self, 
        backstory: str, 
        character_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract claims from a backstory.
        
        Returns list of claim dictionaries with:
        - claim_id: unique identifier
        - text: the claim text
        - category: type of claim
        - importance: estimated importance (high/medium/low)
        - keywords: key terms for retrieval
        """
        if self.use_llm:
            return self._extract_with_llm(backstory, character_name)
        else:
            return self._extract_with_heuristics(backstory, character_name)
    
    def _extract_with_llm(
        self, 
        backstory: str, 
        character_name: Optional[str]
    ) -> List[Dict]:
        """Use LLM to extract structured claims."""
        
        prompt = f"""Analyze this character backstory and extract individual, testable claims.
For each claim, identify:
1. The specific factual assertion
2. Category (early_life, family, education, formative_events, beliefs, fears, ambitions, relationships, personality_traits, physical_attributes, skills_abilities, occupation, location_history)
3. Importance (high = core identity, medium = significant detail, low = minor detail)
4. Key terms that would appear in supporting text

Character name: {character_name or 'Unknown'}

Backstory:
{backstory}

Return a JSON array of claims. Each claim should be:
{{
    "text": "the specific claim",
    "category": "category_name",
    "importance": "high/medium/low",
    "keywords": ["keyword1", "keyword2"]
}}

Extract up to {config.max_claims_per_backstory} most important claims.
Focus on concrete, verifiable facts rather than vague statements.

Return ONLY the JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing character backstories and extracting specific, testable claims."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if content.startswith('```'):
                content = re.sub(r'^```json?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            claims_data = json.loads(content)
            
            # Add claim IDs
            claims = []
            for i, claim in enumerate(claims_data):
                claim['claim_id'] = f"claim_{i}"
                claims.append(claim)
            
            return claims
            
        except Exception as e:
            print(f"LLM extraction failed: {e}, falling back to heuristics")
            return self._extract_with_heuristics(backstory, character_name)
    
    def _extract_with_heuristics(
        self, 
        backstory: str, 
        character_name: Optional[str]
    ) -> List[Dict]:
        """Extract claims using rule-based heuristics."""
        claims = []
        
        # Split into sentences
        sentences = self._split_sentences(backstory)
        
        claim_id = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            # Detect category based on keywords
            category = self._detect_category(sentence)
            importance = self._estimate_importance(sentence, character_name)
            keywords = self._extract_keywords(sentence, character_name)
            
            claims.append({
                'claim_id': f"claim_{claim_id}",
                'text': sentence,
                'category': category,
                'importance': importance,
                'keywords': keywords
            })
            claim_id += 1
        
        # Limit to max claims, prioritizing by importance
        importance_order = {'high': 0, 'medium': 1, 'low': 2}
        claims.sort(key=lambda x: importance_order.get(x['importance'], 2))
        
        return claims[:config.max_claims_per_backstory]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof)\.\s', r'\1<PERIOD> ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore abbreviations
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]
        
        return sentences
    
    def _detect_category(self, sentence: str) -> str:
        """Detect claim category based on keywords."""
        sentence_lower = sentence.lower()
        
        category_keywords = {
            'early_life': ['born', 'childhood', 'grew up', 'young', 'infant', 'baby', 'toddler'],
            'family': ['mother', 'father', 'parent', 'sibling', 'brother', 'sister', 'family', 'married', 'spouse', 'wife', 'husband', 'children', 'son', 'daughter'],
            'education': ['school', 'university', 'college', 'study', 'learn', 'taught', 'degree', 'graduated'],
            'formative_events': ['changed', 'transformed', 'pivotal', 'turning point', 'discovered', 'realized', 'traumatic'],
            'beliefs': ['believe', 'faith', 'value', 'principle', 'ideology', 'philosophy', 'conviction'],
            'fears': ['fear', 'afraid', 'phobia', 'dread', 'terrified', 'nightmare', 'anxiety'],
            'ambitions': ['goal', 'dream', 'ambition', 'aspire', 'hope', 'want', 'desire', 'wish'],
            'relationships': ['friend', 'love', 'hate', 'enemy', 'ally', 'mentor', 'rival'],
            'personality_traits': ['personality', 'character', 'temperament', 'nature', 'disposition'],
            'physical_attributes': ['tall', 'short', 'hair', 'eyes', 'appearance', 'look', 'build', 'scar'],
            'skills_abilities': ['skill', 'ability', 'talent', 'expert', 'proficient', 'master', 'trained'],
            'occupation': ['work', 'job', 'profession', 'career', 'occupation', 'employed'],
            'location_history': ['lived', 'moved', 'relocated', 'home', 'hometown', 'country', 'city']
        }
        
        for category, keywords in category_keywords.items():
            if any(kw in sentence_lower for kw in keywords):
                return category
        
        return 'general'
    
    def _estimate_importance(
        self, 
        sentence: str, 
        character_name: Optional[str]
    ) -> str:
        """Estimate claim importance."""
        sentence_lower = sentence.lower()
        
        # High importance indicators
        high_indicators = [
            'born', 'died', 'killed', 'married', 'only', 'always', 'never',
            'most important', 'defining', 'core', 'fundamental'
        ]
        
        # Check if character name is mentioned (high importance)
        if character_name and character_name.lower() in sentence_lower:
            if any(ind in sentence_lower for ind in high_indicators):
                return 'high'
        
        if any(ind in sentence_lower for ind in high_indicators):
            return 'high'
        
        # Medium importance
        medium_indicators = [
            'often', 'usually', 'sometimes', 'became', 'learned', 'discovered'
        ]
        if any(ind in sentence_lower for ind in medium_indicators):
            return 'medium'
        
        return 'low'
    
    def _extract_keywords(
        self, 
        sentence: str, 
        character_name: Optional[str]
    ) -> List[str]:
        """Extract key terms for retrieval."""
        # Simple keyword extraction
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'was', 'were', 'been', 'be', 'are',
            'it', 'its', 'this', 'that', 'these', 'those', 'he', 'she', 'they',
            'his', 'her', 'their', 'him', 'them', 'who', 'which', 'what', 'when',
            'where', 'how', 'has', 'have', 'had', 'would', 'could', 'should'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower())
        keywords = [w for w in words if w not in stop_words]
        
        # Add character name if provided
        if character_name:
            keywords.insert(0, character_name.lower())
        
        # Return unique keywords
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]