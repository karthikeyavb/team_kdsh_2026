"""
Aggregate claim-level decisions into final story-level predictions.
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter

from .config import config
from .consistency_checker import ConsistencyVerdict


@dataclass
class AggregationResult:
    """Result of aggregating claim-level decisions."""
    story_id: str
    prediction: int  # 1 = consistent, 0 = inconsistent
    confidence: float
    rationale: str
    claim_summary: Dict
    global_check_result: Dict = None


class DecisionAggregator:
    """
    Aggregates claim-level consistency decisions into final prediction.
    """
    
    def __init__(
        self,
        contradiction_threshold: int = None,
        support_ratio_threshold: float = None
    ):
        self.contradiction_threshold = (
            contradiction_threshold or config.contradiction_threshold
        )
        self.support_ratio_threshold = (
            support_ratio_threshold or config.support_ratio_threshold
        )
    
    def aggregate(
        self,
        story_id: str,
        claim_results: List[Dict],
        global_check: Dict = None
    ) -> AggregationResult:
        """
        Aggregate claim results into final prediction.
        
        Decision logic:
        1. If any high-importance claim is contradicted → 0
        2. If multiple claims are contradicted → 0
        3. If global check finds major contradiction → 0
        4. Otherwise, if sufficient claims supported → 1
        5. Default to 1 if no clear contradictions
        """
        # Count verdicts by importance
        verdict_counts = {
            'high': Counter(),
            'medium': Counter(),
            'low': Counter(),
            'all': Counter()
        }
        
        contradictions = []
        supports = []
        
        for result in claim_results:
            verdict = result.get('verdict', 'uncertain')
            importance = result.get('importance', 'medium')
            
            # Handle if importance is in the claim, not result
            if 'claim' in result:
                importance = result['claim'].get('importance', 'medium')
            
            verdict_counts[importance][verdict] += 1
            verdict_counts['all'][verdict] += 1
            
            if verdict == ConsistencyVerdict.CONTRADICTS.value:
                contradictions.append(result)
            elif verdict == ConsistencyVerdict.SUPPORTS.value:
                supports.append(result)
        
        # Calculate metrics
        total_claims = len(claim_results)
        num_contradictions = verdict_counts['all'][ConsistencyVerdict.CONTRADICTS.value]
        num_supports = verdict_counts['all'][ConsistencyVerdict.SUPPORTS.value]
        high_contradictions = verdict_counts['high'][ConsistencyVerdict.CONTRADICTS.value]
        
        # Check global consistency
        global_contradiction = False
        if global_check:
            global_contradiction = global_check.get('has_contradiction', False)
            if global_contradiction:
                major_contradictions = [
                    c for c in global_check.get('contradictions', [])
                    if c.get('severity') == 'major'
                ]
                global_contradiction = len(major_contradictions) > 0
        
        # Make decision
        prediction, confidence, reason = self._make_decision(
            num_contradictions=num_contradictions,
            num_supports=num_supports,
            high_contradictions=high_contradictions,
            total_claims=total_claims,
            global_contradiction=global_contradiction,
            contradictions=contradictions
        )
        
        # Generate rationale
        rationale = self._generate_rationale(
            prediction=prediction,
            reason=reason,
            contradictions=contradictions,
            supports=supports,
            global_check=global_check
        )
        
        return AggregationResult(
            story_id=story_id,
            prediction=prediction,
            confidence=confidence,
            rationale=rationale,
            claim_summary={
                'total': total_claims,
                'supported': num_supports,
                'contradicted': num_contradictions,
                'not_mentioned': verdict_counts['all'][ConsistencyVerdict.NOT_MENTIONED.value],
                'uncertain': verdict_counts['all'][ConsistencyVerdict.UNCERTAIN.value]
            },
            global_check_result=global_check
        )
    
    def _make_decision(
        self,
        num_contradictions: int,
        num_supports: int,
        high_contradictions: int,
        total_claims: int,
        global_contradiction: bool,
        contradictions: List[Dict]
    ) -> Tuple[int, float, str]:
        """
        Make the final decision.
        Returns (prediction, confidence, reason).
        """
        # Rule 1: High-importance contradiction
        if high_contradictions > 0:
            return (0, 0.85, "high_importance_contradiction")
        
        # Rule 2: Global contradiction
        if global_contradiction:
            return (0, 0.80, "global_contradiction")
        
        # Rule 3: Multiple contradictions
        if num_contradictions >= self.contradiction_threshold:
            confidence = min(0.9, 0.6 + num_contradictions * 0.1)
            return (0, confidence, "multiple_contradictions")
        
        # Rule 4: Strong contradictions (high confidence in the contradiction)
        strong_contradictions = [
            c for c in contradictions 
            if c.get('confidence', 0) > 0.8
        ]
        if len(strong_contradictions) > 0:
            return (0, 0.75, "strong_contradiction")
        
        # Rule 5: Check support ratio
        if total_claims > 0:
            support_ratio = num_supports / total_claims
            if support_ratio >= self.support_ratio_threshold:
                confidence = 0.6 + support_ratio * 0.3
                return (1, min(0.95, confidence), "sufficient_support")
        
        # Default: Consistent if no clear contradictions
        return (1, 0.55, "no_clear_contradictions")
    
    def _generate_rationale(
        self,
        prediction: int,
        reason: str,
        contradictions: List[Dict],
        supports: List[Dict],
        global_check: Dict = None
    ) -> str:
        """Generate a human-readable rationale."""
        
        if prediction == 0:
            # Inconsistent
            if reason == "high_importance_contradiction":
                if contradictions:
                    c = contradictions[0]
                    return f"Critical contradiction found: '{c.get('claim_text', 'Unknown claim')[:50]}...' conflicts with novel evidence."
                return "Found contradiction in a core character trait."
            
            elif reason == "global_contradiction":
                if global_check and global_check.get('contradictions'):
                    c = global_check['contradictions'][0]
                    return f"Backstory conflicts with novel: {c.get('backstory_claim', '')[:40]} vs. novel's {c.get('novel_fact', '')[:40]}"
                return "Global character facts from novel contradict backstory."
            
            elif reason == "multiple_contradictions":
                return f"Found {len(contradictions)} contradicting claims across the backstory."
            
            elif reason == "strong_contradiction":
                if contradictions:
                    c = contradictions[0]
                    return f"Clear contradiction: '{c.get('claim_text', '')[:50]}...' - {c.get('reasoning', '')[:50]}"
                return "Found clear contradiction with high confidence."
            
            return "Backstory contains inconsistencies with the novel."
        
        else:
            # Consistent
            if reason == "sufficient_support":
                return f"Backstory aligns with novel: {len(supports)} claims supported, no clear contradictions."
            
            elif reason == "no_clear_contradictions":
                return "No contradictions found; backstory is compatible with the novel's narrative."
            
            return "Backstory is consistent with the novel."


class FeatureBasedAggregator:
    """
    Alternative aggregator using feature-based classification.
    Can be trained on labeled data for better performance.
    """
    
    def __init__(self):
        self.weights = {
            'contradiction_count': -0.4,
            'support_count': 0.2,
            'high_contradiction': -0.8,
            'support_ratio': 0.3,
            'coverage': 0.1,
            'avg_confidence': 0.2,
            'global_contradiction': -0.5
        }
        self.threshold = 0.0
    
    def extract_features(
        self,
        claim_results: List[Dict],
        global_check: Dict = None
    ) -> Dict[str, float]:
        """Extract features from claim results."""
        total = len(claim_results) or 1
        
        contradictions = [r for r in claim_results if r.get('verdict') == 'contradicts']
        supports = [r for r in claim_results if r.get('verdict') == 'supports']
        
        high_contradictions = sum(
            1 for r in contradictions 
            if r.get('importance', 'medium') == 'high'
        )
        
        avg_confidence = sum(
            r.get('confidence', 0.5) for r in claim_results
        ) / total
        
        coverage = sum(
            1 for r in claim_results 
            if r.get('verdict') not in ['not_mentioned', 'uncertain']
        ) / total
        
        return {
            'contradiction_count': len(contradictions),
            'support_count': len(supports),
            'high_contradiction': high_contradictions,
            'support_ratio': len(supports) / total,
            'coverage': coverage,
            'avg_confidence': avg_confidence,
            'global_contradiction': 1.0 if (global_check and global_check.get('has_contradiction')) else 0.0
        }
    
    def predict(
        self,
        claim_results: List[Dict],
        global_check: Dict = None
    ) -> Tuple[int, float]:
        """Make prediction using weighted features."""
        features = self.extract_features(claim_results, global_check)
        
        score = sum(
            self.weights.get(k, 0) * v 
            for k, v in features.items()
        )
        
        # Convert to probability-like confidence
        import math
        probability = 1 / (1 + math.exp(-score))
        
        prediction = 1 if score > self.threshold else 0
        confidence = probability if prediction == 1 else (1 - probability)
        
        return prediction, confidence