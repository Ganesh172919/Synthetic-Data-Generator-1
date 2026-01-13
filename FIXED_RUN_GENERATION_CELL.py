"""
FIXED RUN_GENERATION FUNCTION - Copy this to your notebook Cell 9
Replace the entire run_generation function with this version.
"""

# ============================================================================
# CELL 9: MAIN EXECUTION ENGINE (FIXED VERSION)
# ============================================================================

def run_generation(user_request: str = None, target_samples: int = None):
    """
    Main function to run the synthetic data generation pipeline.

    Args:
        user_request: Natural language description of what to generate
        target_samples: Number of samples to generate (default from config)

    Returns:
        Tuple of (DataFrame, stats_dict, csv_path)
    """

    if target_samples:
        config.TARGET_SAMPLES = target_samples

    if user_request is None:
        user_request = f"""
        Generate {config.TARGET_SAMPLES} high-quality financial question-answer pairs.
        Cover all major financial categories including investing, banking, taxation,
        retirement planning, personal finance, stock markets, and risk management.
        Include a mix of difficulty levels from beginner to expert.
        Ensure practical, educational content suitable for training AI models.
        """

    console.print(Panel.fit(
        f"Starting SynthAgent Engine\n"
        f"   Target: {config.TARGET_SAMPLES} samples\n"
        f"   Model: {config.MODEL_ID}\n"
        f"   Quality threshold: {config.MIN_QUALITY_SCORE}/10",
        title="Generation Started"
    ))

    # Create workflow
    workflow, aggregator = create_synth_workflow(llm)

    # Initialize state with proper default values (use empty dicts, not None)
    initial_state = {
        "user_input": user_request,
        "requirements": {},
        "schema": {},
        "context": {},
        "current_batch": {},
        "generated_samples": [],
        "quality_scores": [],
        "total_generated": 0,
        "total_accepted": 0,
        "errors": [],
        "current_step": "init",
        "hitl_pause": False,
        "final_output": "",
        "consecutive_failed_batches": 0
    }

    # Run with progress tracking - set reasonable recursion limit
    config_run = {"recursion_limit": 10000}

    start_time = datetime.now()
    last_progress = 0

    try:
        # Stream execution for progress updates
        for event in workflow.stream(initial_state, config_run):
            # Get current state
            current_accepted = 0
            for node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    current_accepted = node_state.get("total_accepted", 0)
                    current_step = node_state.get("current_step", "")

            # Progress update every 100 samples
            if current_accepted - last_progress >= 100:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = current_accepted / elapsed if elapsed > 0 else 0
                eta = (config.TARGET_SAMPLES - current_accepted) / rate if rate > 0 else 0

                print(f"   Progress: {current_accepted}/{config.TARGET_SAMPLES} "
                      f"({100*current_accepted/config.TARGET_SAMPLES:.1f}%) | "
                      f"Rate: {rate:.1f}/sec | ETA: {eta/60:.1f} min")
                last_progress = current_accepted

            # Memory cleanup periodically
            if current_accepted % 500 == 0 and current_accepted > 0:
                gc.collect()
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()

    # Get final results
    elapsed = (datetime.now() - start_time).total_seconds()
    df = aggregator.to_dataframe()
    stats = aggregator.get_stats()

    # Calculate rate safely (avoid division by zero)
    rate = len(df) / elapsed if elapsed > 0 else 0.0
    avg_quality = stats.get('avg_quality', 0) if stats else 0
    
    # Display summary
    console.print(Panel.fit(
        f"Generation Complete!\n\n"
        f"Statistics:\n"
        f"   Total samples: {len(df)}\n"
        f"   Time elapsed: {elapsed/60:.1f} minutes\n"
        f"   Rate: {rate:.2f} samples/second\n"
        f"   Avg quality: {avg_quality:.2f}/10",
        title="Complete", style="green"
    ))

    # Show category distribution
    if stats and "by_category" in stats and len(df) > 0:
        print("\n" + "="*60)
        print("CATEGORY DISTRIBUTION")
        print("="*60)
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            pct = 100 * count / len(df) if len(df) > 0 else 0
            bar = "*" * int(pct / 2)
            print(f"{cat:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Show difficulty distribution
    print(f"\n{'='*60}")
    print("DIFFICULTY DISTRIBUTION")
    print(f"{'='*60}")
    if stats and "by_difficulty" in stats and len(df) > 0:
        for diff, count in stats["by_difficulty"].items():
            pct = 100 * count / len(df) if len(df) > 0 else 0
            bar = "*" * int(pct / 2)
            print(f"{diff:15s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Export if we have samples
    csv_path = None
    if len(df) > 0:
        exporter = ExporterAgent(config.OUTPUT_DIR)
        csv_path = exporter.export_csv(df, config.CSV_FILENAME)
        exporter.export_metadata(stats)
        print(f"\nOutput saved to: {csv_path}")

    return df, stats, csv_path


def preview_samples(df: pd.DataFrame, n: int = 5):
    """Preview generated samples"""
    if len(df) == 0:
        print("No samples to preview")
        return
        
    console.print(Panel.fit(f"Sample Preview (showing {min(n, len(df))} examples)", title="Preview"))

    for i, row in df.head(n).iterrows():
        print(f"\n{'-'*60}")
        print(f"[{row.get('category', 'N/A')} | {row.get('difficulty', 'N/A')}]")
        print(f"Q: {row['question']}")
        print(f"A: {row['answer'][:200]}..." if len(str(row['answer'])) > 200 else f"A: {row['answer']}")


print("Execution engine ready (FIXED VERSION)")
