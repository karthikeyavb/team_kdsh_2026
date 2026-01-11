"""
LLM-based consistency checking between claims and novel evidence.
"""
import json
import re
from typing import List, Dict, Tuple, Optional
from enum import Enum
from openai import OpenAI

from .config import config


class ConsistencyVerdict(Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NOT_MENTIONED = "not_mentioned"
    UNCERTAIN = "uncertain"


class ConsistencyChecker:
    """
    Checks consistency between backstory claims and novel evidence.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
    
    def check_claim(
        self,
        claim: Dict,
        retrieved_chunks: List[Dict],
        character_name: Optional[str] = None
    ) -> Dict:
        """
        Check if a claim is consistent with retrieved evidence.
        
        Returns:
        - verdict: supports/contradicts/not_mentioned/uncertain
        - confidence: 0.0-1.0
        - evidence: supporting quote or explanation
        - reasoning: brief explanation
        """
        if not retrieved_chunks:
            return {
                'claim_id': claim['claim_id'],
                'claim_text': claim['text'],
                'verdict': ConsistencyVerdict.NOT_MENTIONED.value,
                'confidence': 0.5,
                'evidence': None,
                'reasoning': 'No relevant passages found in the novel.'
            }
        
        # Prepare evidence text
        evidence_text = self._prepare_evidence(retrieved_chunks)
        
        # Use LLM to analyze
        result = self._analyze_with_llm(claim, evidence_text, character_name)
        
        return result
    
    def _prepare_evidence(self, chunks: List[Dict]) -> str:
        """Prepare evidence text from retrieved chunks."""
        evidence_parts = []
        
        for i, chunk_info in enumerate(chunks[:5]):  # Limit to top 5
            chunk = chunk_info.get('chunk', chunk_info)
            content = chunk_info.get('content', chunk.get('content', ''))
            score = chunk_info.get('score', 0)
            
            # Truncate very long chunks
            if len(content) > 1500:
                content = content[:1500] + "..."
            
            evidence_parts.append(f"[Passage {i+1}, relevance: {score:.2f}]\n{content}")
        
        return "\n\n---\n\n".join(evidence_parts)
    
    def _analyze_with_llm(
        self,
        claim: Dict,
        evidence_text: str,
        character_name: Optional[str]
    ) -> Dict:
        """Use LLM to analyze consistency."""
        
        prompt = f"""You are analyzing whether a character backstory claim is consistent with evidence from a novel.

Character: {character_name or 'Unknown'}

CLAIM TO VERIFY:
"{claim['text']}"
Category: {claim.get('category', 'unknown')}

EVIDENCE FROM NOVEL:
{evidence_text}

Analyze carefully and determine:
1. Does the novel SUPPORT this claim (evidence confirms it)?
2. Does the novel CONTRADICT this claim (evidence conflicts with it)?
3. Is the claim NOT MENTIONED (no relevant evidence either way)?
4. Is the verdict UNCERTAIN (ambiguous or partial evidence)?

Consider:
- Direct statements vs. implied information
- Chronology and timing
- Character actions that support or contradict stated traits/beliefs
- Consistency with described relationships and events

Respond with a JSON object:
{{
    "verdict": "supports" | "contradicts" | "not_mentioned" | "uncertain",
    "confidence": 0.0 to 1.0,
    "evidence": "relevant quote or description from the passages (if any)",
    "reasoning": "brief explanation of your judgment"
}}

Return ONLY the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "system", "content": "You are an expert literary analyst specializing in character consistency analysis. Be precise and evidence-based in your judgments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.llm_temperature,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown
            if content.startswith('```'):
                content = re.sub(r'^```json?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            result = json.loads(content)
            
            # Validate verdict
            valid_verdicts = [v.value for v in ConsistencyVerdict]
            if result.get('verdict') not in valid_verdicts:
                result['verdict'] = 'uncertain'
            
            # Add claim info
            result['claim_id'] = claim['claim_id']
            result['claim_text'] = claim['text']
            
            return result
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return {
                'claim_id': claim['claim_id'],
                'claim_text': claim['text'],
                'verdict': ConsistencyVerdict.UNCERTAIN.value,
                'confidence': 0.3,
                'evidence': None,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    def check_all_claims(
        self,
        claims: List[Dict],
        retriever_func,  # Function to retrieve chunks for a claim
        story_id: str,
        character_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Check all claims for a backstory.
        """
        results = []
        
        for claim in claims:
            # Retrieve relevant chunks
            retrieved = retriever_func(claim['text'], story_id)
            
            # Check consistency
            result = self.check_claim(claim, retrieved, character_name)
            results.append(result)
        
        return results


class GlobalConsistencyChecker:
    """
    Additional checks for global consistency using character tracking.
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
    
    def extract_character_facts(
        self, 
        chunks: List[Dict], 
        character_name: str
    ) -> Dict:
        """
        Extract key facts about a character from novel chunks.
        """
        # Combine relevant chunks
        relevant_text = ""
        for chunk in chunks:
            content = chunk.get('content', '')
            if character_name.lower() in content.lower():
                relevant_text += content + "\n\n"
                if len(relevant_text) > 10000:
                    break
        
        if not relevant_text:
            return {}
        
        prompt = f"""Extract key biographical facts about the character "{character_name}" from these passages:

{relevant_text[:8000]}

Return a JSON object with known facts:
{{
    "birthplace": "location or null",
    "family_members": ["list of known family"],
    "occupation": "known occupation or null",
    "key_events": ["list of major life events mentioned"],
    "personality_traits": ["observed traits"],
    "relationships": ["key relationships mentioned"],
    "beliefs_values": ["stated or demonstrated beliefs"]
}}

Only include facts that are clearly stated or strongly implied. Use null for unknown facts.
Return ONLY the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = re.sub(r'^```json?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Fact extraction failed: {e}")
            return {}
    
    def check_global_consistency(
        self,
        backstory: str,
        character_facts: Dict,
        character_name: str
    ) -> Dict:
        """
        Check global consistency between backstory and extracted facts.
        """
        if not character_facts:
            return {
                'has_global_contradiction': False,
                'issues': [],
                'confidence': 0.3
            }
        
        prompt = f"""Compare this character backstory with known facts from the novel.

CHARACTER: {character_name}

BACKSTORY:
{backstory}

KNOWN FACTS FROM NOVEL:
{json.dumps(character_facts, indent=2)}

Identify any CONTRADICTIONS between the backstory and the known facts.
Only flag clear contradictions, not missing information.

Return a JSON object:
{{
    "has_contradiction": true/false,
    "contradictions": [
        {{
            "backstory_claim": "what the backstory says",
            "novel_fact": "what the novel says",
            "severity": "major" or "minor"
        }}
    ],
    "confidence": 0.0 to 1.0
}}

Return ONLY the JSON object."""

        try:
            response = self.client.chat.completions.create(
                model=config.llm_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```'):
                content = re.sub(r'^```json?\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Global consistency check failed: {e}")
            return {
                'has_contradiction': False,
                'contradictions': [],
                'confidence': 0.3
            }