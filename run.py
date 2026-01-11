#!/usr/bin/env python3
"""
Main entry point for running the consistency checker.
"""
import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main, ConsistencyPipeline, load_inputs_from_directory, BackstoryInput
from src.config import config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Novel-Backstory Consistency Checker"
    )
    
    parser.add_argument(
        "--novels-dir",
        type=str,
        default="data/novels",
        help="Directory containing novel .txt files"
    )
    
    parser.add_argument(
        "--backstories-dir",
        type=str,
        default="data/backstories",
        help="Directory containing backstory files"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/results.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Process single story ID only"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Chunk size in tokens"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per claim"
    )
    
    return parser.parse_args()


def run():
    args = parse_args()
    
    # Update config
    config.novels_dir = args.novels_dir
    config.backstories_dir = args.backstories_dir
    config.chunk_size = args.chunk_size
    config.top_k_chunks = args.top_k
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.single:
        # Process single story
        pipeline = ConsistencyPipeline()
        pipeline.initialize()
        
        inputs = load_inputs_from_directory(
            config.novels_dir,
            config.backstories_dir
        )
        
        target = [i for i in inputs if i.story_id == args.single]
        if not target:
            print(f"Story ID '{args.single}' not found")
            return
        
        results = pipeline.process_batch(target)
        pipeline.save_results(results, args.output)
    else:
        # Process all
        main()


if __name__ == "__main__":
    run()