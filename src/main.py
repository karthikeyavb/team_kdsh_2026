"""
Main orchestration module for the consistency checking pipeline.
"""
import os
import csv
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .config import config, Config
from .config import config, Config
from .claim_extractor import ClaimExtractor
from .consistency_checker import ConsistencyChecker, GlobalConsistencyChecker
from .aggregator import DecisionAggregator, AggregationResult

# Try to import Pathway pipeline, fallback to simple if failed
try:
    from .pathway_pipeline import PathwayRAGPipeline
    USE_PATHWAY = True
except (ImportError, ModuleNotFoundError):
    from .simple_pipeline import SimpleRAGPipeline as PathwayRAGPipeline
    USE_PATHWAY = False
    print("WARNING: 'pathway' module not found. Using SimpleRAGPipeline (Windows Fallback).")


@dataclass
class BackstoryInput:
    """Input data for a backstory to check."""
    story_id: str
    novel_path: str
    backstory_text: str
    character_name: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class PredictionOutput:
    """Output prediction for a backstory."""
    story_id: str
    prediction: int
    rationale: str
    confidence: float
    details: Optional[Dict] = None


class ConsistencyPipeline:
    """
    Main pipeline orchestrating all components.
    """
    
    def __init__(self, config_obj: Config = None):
        self.config = config_obj or config
        self.config.validate()
        
        # Initialize components
        if USE_PATHWAY:
            self.rag_pipeline = PathwayRAGPipeline()
        else:
            self.rag_pipeline = PathwayRAGPipeline()  # This is actually SimpleRAGPipeline aliased
            
        self.claim_extractor = ClaimExtractor(use_llm=True)
        self.consistency_checker = ConsistencyChecker()
        self.global_checker = GlobalConsistencyChecker()
        self.aggregator = DecisionAggregator()
        
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the pipeline (process novels, build index)."""
        print("Initializing Consistency Pipeline...")
        print("=" * 50)
        
        if not USE_PATHWAY:
            print("NOTE: Running in Windows Compatibility Mode (No Pathway)")
        
        # Initialize RAG pipeline
        self.rag_pipeline.initialize()
        
        self.is_initialized = True
        print("=" * 50)
        print("Pipeline initialized successfully!\n")
    
    def process_single(
        self,
        backstory_input: BackstoryInput
    ) -> PredictionOutput:
        """
        Process a single backstory and return prediction.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        story_id = backstory_input.story_id
        backstory = backstory_input.backstory_text
        character_name = backstory_input.character_name
        
        print(f"\nProcessing story: {story_id}")
        
        # Step 1: Extract claims from backstory
        print("  Extracting claims...")
        claims = self.claim_extractor.extract_claims(backstory, character_name)
        print(f"  Found {len(claims)} claims")
        
        # Step 2: Check each claim against novel
        print("  Checking claims against novel...")
        claim_results = []
        
        for claim in claims:
            # Retrieve relevant chunks
            retrieved = self.rag_pipeline.retrieve_for_claim(
                claim['text'],
                story_id,
                top_k=self.config.top_k_chunks
            )
            
            # Check consistency
            result = self.consistency_checker.check_claim(
                claim,
                retrieved,
                character_name
            )
            result['importance'] = claim.get('importance', 'medium')
            claim_results.append(result)
        
        # Step 3: Optional global consistency check
        print("  Running global consistency check...")
        story_chunks = self.rag_pipeline.get_story_chunks(story_id)
        
        global_check = None
        if character_name and story_chunks:
            # Extract character facts
            char_facts = self.global_checker.extract_character_facts(
                story_chunks,
                character_name
            )
            
            # Check global consistency
            global_check = self.global_checker.check_global_consistency(
                backstory,
                char_facts,
                character_name
            )
        
        # Step 4: Aggregate to final decision
        print("  Aggregating results...")
        result = self.aggregator.aggregate(
            story_id=story_id,
            claim_results=claim_results,
            global_check=global_check
        )
        
        print(f"  Prediction: {result.prediction} (confidence: {result.confidence:.2f})")
        print(f"  Rationale: {result.rationale}")
        
        return PredictionOutput(
            story_id=story_id,
            prediction=result.prediction,
            rationale=result.rationale,
            confidence=result.confidence,
            details={
                'claim_summary': result.claim_summary,
                'claim_results': claim_results,
                'global_check': global_check
            }
        )
    
    def process_batch(
        self,
        inputs: List[BackstoryInput]
    ) -> List[PredictionOutput]:
        """Process multiple backstories."""
        results = []
        
        for i, input_data in enumerate(inputs):
            print(f"\n[{i+1}/{len(inputs)}] Processing {input_data.story_id}")
            
            try:
                result = self.process_single(input_data)
                results.append(result)
            except Exception as e:
                print(f"  Error processing {input_data.story_id}: {e}")
                # Return default prediction on error
                results.append(PredictionOutput(
                    story_id=input_data.story_id,
                    prediction=1,  # Default to consistent
                    rationale=f"Processing error: {str(e)[:50]}",
                    confidence=0.0
                ))
        
        return results
    
    def save_results(
        self,
        results: List[PredictionOutput],
        output_path: str = None
    ):
        """Save results to CSV file."""
        output_path = output_path or os.path.join(
            self.config.output_dir, 
            "results.csv"
        )
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Story ID', 'Prediction', 'Rationale'])
            
            for result in results:
                writer.writerow([
                    result.story_id,
                    result.prediction,
                    result.rationale
                ])
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save detailed JSON
        json_path = output_path.replace('.csv', '_detailed.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [asdict(r) for r in results],
                f,
                indent=2,
                default=str
            )
        print(f"Detailed results saved to: {json_path}")


def load_inputs_from_directory(
    novels_dir: str,
    backstories_dir: str
) -> List[BackstoryInput]:
    """
    Load inputs from directories.
    Expects:
    - novels_dir: contains {story_id}.txt files
    - backstories_dir: contains {story_id}_backstory.txt or {story_id}.json files
    """
    inputs = []
    
    # Get all novel files
    novel_files = [f for f in os.listdir(novels_dir) if f.endswith('.txt')]
    
    for novel_file in novel_files:
        story_id = os.path.splitext(novel_file)[0]
        novel_path = os.path.join(novels_dir, novel_file)
        
        # Look for corresponding backstory
        backstory_txt = os.path.join(backstories_dir, f"{story_id}_backstory.txt")
        backstory_json = os.path.join(backstories_dir, f"{story_id}.json")
        
        backstory_text = None
        character_name = None
        
        if os.path.exists(backstory_txt):
            with open(backstory_txt, 'r', encoding='utf-8') as f:
                backstory_text = f.read()
        elif os.path.exists(backstory_json):
            with open(backstory_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                backstory_text = data.get('backstory', '')
                character_name = data.get('character_name')
        
        if backstory_text:
            inputs.append(BackstoryInput(
                story_id=story_id,
                novel_path=novel_path,
                backstory_text=backstory_text,
                character_name=character_name
            ))
    
    return inputs


def main():
    """Main entry point."""
    print("=" * 60)
    print("Novel-Backstory Consistency Checker")
    print("Track A Submission - KDSH 2026")
    print("=" * 60)
    
    # Load inputs
    print("\nLoading inputs...")
    inputs = load_inputs_from_directory(
        config.novels_dir,
        config.backstories_dir
    )
    print(f"Found {len(inputs)} story-backstory pairs")
    
    if not inputs:
        print("No inputs found! Please check your data directories.")
        return
    
    # Initialize pipeline
    pipeline = ConsistencyPipeline()
    pipeline.initialize()
    
    # Process all inputs
    print("\n" + "=" * 60)
    print("Processing backstories...")
    print("=" * 60)
    
    results = pipeline.process_batch(inputs)
    
    # Save results
    pipeline.save_results(results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    consistent = sum(1 for r in results if r.prediction == 1)
    inconsistent = sum(1 for r in results if r.prediction == 0)
    print(f"Total processed: {len(results)}")
    print(f"Consistent (1): {consistent}")
    print(f"Inconsistent (0): {inconsistent}")
    print("=" * 60)


if __name__ == "__main__":
    main()