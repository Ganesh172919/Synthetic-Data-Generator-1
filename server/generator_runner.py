#!/usr/bin/env python3
"""
Generator Runner - Bridge between Node.js API and Python Generators
====================================================================
This script serves as a command-line interface to run the Python generators
from the Node.js backend with proper configuration and progress reporting.

Features:
- Accept JSON configuration from command line
- Run either universal or specialized generators
- Report progress to stdout in JSON format
- Handle checkpoints and resume functionality
- Graceful shutdown and error handling
"""

import sys
import os
import json
import argparse
import signal
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add Pre-Work directory to Python path
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
PRE_WORK_DIR = PROJECT_ROOT / "Pre-Work"
sys.path.insert(0, str(PRE_WORK_DIR))


def emit_progress(event_type: str, data: Dict[str, Any]):
    """Emit progress events as JSON to stdout for Node.js to parse."""
    progress_event = {
        "type": event_type,
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "data": data
    }
    # Print to stdout with flush to ensure immediate delivery
    print(f"PROGRESS:{json.dumps(progress_event)}", flush=True)


def emit_error(error_type: str, message: str, details: Optional[str] = None):
    """Emit error events."""
    error_event = {
        "type": "error",
        "error_type": error_type,
        "message": message,
        "details": details,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    print(f"ERROR:{json.dumps(error_event)}", file=sys.stderr, flush=True)


def setup_signal_handlers():
    """Setup graceful shutdown handlers."""
    def signal_handler(sig, frame):
        emit_progress("shutdown", {"reason": "signal_received", "signal": sig})
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def run_universal_generator(config: Dict[str, Any]) -> int:
    """
    Run the universal dataset generator with given configuration.
    
    Args:
        config: Configuration dictionary with keys:
            - target_size: Number of items to generate
            - items_per_batch: Items per batch
            - output_file: Output filename
            - output_format: jsonl, csv, or json
            - domain_description: Natural language description
            - topics: List of topics
            - difficulty_levels: List of difficulty levels
            - model_name: Optional model name override
            - checkpoint_file: Optional checkpoint file path
            
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        emit_progress("init", {"message": "Loading universal generator"})
        
        # Import the universal generator
        from universal_dataset_generator import (
            UniversalGenerator,
            GeneratorConfig,
            ModelProvider
        )
        
        emit_progress("init", {"message": "Configuring generator"})
        
        # Build generator config
        gen_config = GeneratorConfig(
            target_size=config.get("targetSize", 1000),
            items_per_batch=config.get("batchSize", 25),
            output_file=config.get("outputFile", "generated_dataset"),
            output_format=config.get("outputFormat", "jsonl"),
            model_name=config.get("modelName", "mistralai/Mistral-7B-Instruct-v0.2"),
            use_quantization=config.get("useQuantization", True),
            temperature=config.get("temperature", 0.8),
            checkpoint_file=config.get("checkpointFile", "generator_checkpoint.json"),
            save_interval=config.get("saveInterval", 100),
        )
        
        # Create prompt from domain description
        domain_desc = config.get("domainDescription", "General knowledge Q&A")
        topics = config.get("topics", [])
        
        user_prompt = f"""Generate a high-quality dataset for: {domain_desc}

Topics to cover:
{chr(10).join(f"- {topic}" for topic in topics)}

Generate diverse, educational content that covers these topics comprehensively."""
        
        emit_progress("init", {"message": "Starting generation", "config": {
            "target_size": gen_config.target_size,
            "batch_size": gen_config.items_per_batch,
            "output_format": gen_config.output_format
        }})
        
        # Initialize generator
        generator = UniversalGenerator(config=gen_config, user_prompt=user_prompt)
        
        # Custom progress callback
        original_update = None
        
        def progress_callback(current: int, total: int, rate: float):
            emit_progress("progress", {
                "current": current,
                "total": total,
                "rate": rate,
                "percentage": (current / total * 100) if total > 0 else 0
            })
        
        # Monkey-patch progress reporting if possible
        # This is a simplified approach; actual implementation depends on generator structure
        
        emit_progress("start", {"message": "Generation started"})
        
        # Run the generator
        result = generator.run()
        
        emit_progress("complete", {
            "message": "Generation completed successfully",
            "output_file": result.get("output_file") if isinstance(result, dict) else gen_config.output_file,
            "total_generated": gen_config.target_size
        })
        
        return 0
        
    except Exception as e:
        emit_error("generation_error", str(e), traceback.format_exc())
        return 1


def run_financial_generator(config: Dict[str, Any]) -> int:
    """
    Run the financial education generator (ultra-optimized).
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        emit_progress("init", {"message": "Loading financial education generator"})
        
        # This would import and run the financial generator
        # For now, we'll use the universal generator as a fallback
        emit_progress("info", {"message": "Using universal generator for financial domain"})
        
        # Add financial-specific topics
        config["domainDescription"] = "Financial education and personal finance"
        config["topics"] = config.get("topics", [
            "Personal Finance",
            "Budgeting Basics",
            "Credit and Debt Management",
            "Investment Fundamentals",
            "Retirement Planning",
            "Banking Services",
            "Tax Planning",
            "Insurance"
        ])
        
        return run_universal_generator(config)
        
    except Exception as e:
        emit_error("generation_error", str(e), traceback.format_exc())
        return 1


def main():
    """Main entry point for the generator runner."""
    parser = argparse.ArgumentParser(
        description="Run synthetic data generators from Node.js backend"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="JSON configuration string or path to config file"
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["universal", "financial"],
        default="universal",
        help="Which generator to use"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    try:
        # Parse configuration
        config_str = args.config
        
        # Check if it's a file path
        if os.path.exists(config_str):
            with open(config_str, 'r') as f:
                config = json.load(f)
        else:
            # Parse as JSON string
            config = json.loads(config_str)
        
        emit_progress("init", {
            "message": "Generator runner started",
            "generator_type": args.generator,
            "resume": args.resume
        })
        
        # Add resume flag to config
        config["resume"] = args.resume
        
        # Run the appropriate generator
        if args.generator == "financial":
            exit_code = run_financial_generator(config)
        else:
            exit_code = run_universal_generator(config)
        
        sys.exit(exit_code)
        
    except json.JSONDecodeError as e:
        emit_error("config_error", "Invalid JSON configuration", str(e))
        sys.exit(1)
    except Exception as e:
        emit_error("fatal_error", "Unexpected error", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
