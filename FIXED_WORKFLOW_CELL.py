"""
FIXED LANGGRAPH WORKFLOW - Copy this entire code block to your notebook Cell 8
Replace the entire create_synth_workflow function with this version.

This fixes the 'list' object is not a mapping error and other issues.
"""

# ============================================================================
# CELL 8: LANGGRAPH WORKFLOW (FIXED VERSION)
# ============================================================================

def create_synth_workflow(llm):
    """Create the LangGraph workflow for synthetic data generation"""

    # Initialize agents
    parser = RequirementParserAgent(llm)
    context_builder = ContextBuilderAgent(llm)
    generator = MasterDataGeneratorAgent(llm)
    quality_controller = QualityControllerAgent(llm, threshold=config.MIN_QUALITY_SCORE)
    aggregator = DatasetAggregator()
    exporter = ExporterAgent(config.OUTPUT_DIR)

    # Define workflow nodes - IMPORTANT: Return dict with updated keys only (LangGraph v0.2+ requirement)
    def parse_requirements(state) -> Dict:
        """Node: Parse user requirements"""
        print("\nParsing requirements...")
        user_input = state.get("user_input", "Generate financial Q&A dataset")
        requirements = parser.invoke(user_input)
        print(f"   Domain: {requirements.get('domain')}")
        print(f"   Target: {requirements.get('target_size')} samples")
        # Return only the updated keys - LangGraph will merge them
        return {
            "requirements": requirements,
            "current_step": "requirements_parsed"
        }

    def build_context(state) -> Dict:
        """Node: Build domain context for current batch"""
        categories = list(FINANCIAL_KNOWLEDGE["categories"].keys())

        # Get requirements from state
        requirements = state.get("requirements", {})
        
        # Select random category and difficulty based on distribution
        category = random.choice(categories)
        diff_dist = requirements.get("difficulty_distribution",
                                     {"beginner": 0.25, "intermediate": 0.40,
                                      "advanced": 0.25, "expert": 0.10})
        difficulty = random.choices(
            list(diff_dist.keys()),
            weights=list(diff_dist.values())
        )[0]

        context = context_builder.invoke(category, difficulty)
        new_context = {
            "category": category,
            "difficulty": difficulty,
            **context
        }
        # Return only the updated keys
        return {
            "context": new_context,
            "current_step": "context_built"
        }

    def generate_batch(state) -> Dict:
        """Node: Generate a batch of Q&A pairs"""
        context = state.get("context", {})
        category = context.get("category", "investing")
        difficulty = context.get("difficulty", "intermediate")

        samples = generator.invoke(
            category=category,
            difficulty=difficulty,
            context=context,
            num_samples=config.SAMPLES_PER_LLM_CALL
        )

        current_batch = state.get("current_batch") or {}
        new_batch = {
            "samples": samples,
            "category": category,
            "difficulty": difficulty,
            "correction_rounds": current_batch.get("correction_rounds", 0)
        }
        # Return only the updated keys
        return {
            "current_batch": new_batch,
            "current_step": "batch_generated"
        }

    def check_quality(state) -> Dict:
        """Node: Quality control for current batch"""
        batch = dict(state.get("current_batch", {}))  # Make a copy
        samples = batch.get("samples", [])

        if samples:
            score = quality_controller.invoke(samples)
            batch["quality_score"] = score.model_dump()
            batch["passed"] = score.passed
        else:
            batch["passed"] = False
            batch["quality_score"] = {"overall": 0, "feedback": "No samples generated"}

        # Return only the updated keys
        return {
            "current_batch": batch,
            "current_step": "quality_checked"
        }

    def should_retry(state) -> str:
        """Conditional edge: Decide if batch needs retry."""
        batch = state.get("current_batch", {})
        passed = batch.get("passed", False)
        rounds = batch.get("correction_rounds", 0)
        consecutive_failures = state.get("consecutive_failed_batches", 0)

        if passed:
            return "aggregate"
        else:
            new_consecutive = consecutive_failures + 1
            if new_consecutive >= config.MAX_CONSECUTIVE_FAILURES:
                return "stop_generation"
            elif rounds >= config.MAX_CORRECTION_ROUNDS:
                return "aggregate"
            else:
                return "retry"

    def aggregate_results(state) -> Dict:
        """Node: Aggregate accepted samples"""
        batch = state.get("current_batch", {})
        samples = batch.get("samples", [])
        errors = list(state.get("errors", []))  # Copy the errors list
        
        total_accepted = state.get("total_accepted", 0)
        consecutive_failed = state.get("consecutive_failed_batches", 0)
        hitl_pause = False

        if batch.get("passed", False) and samples:
            score_dict = batch.get("quality_score", {})
            score = QualityScore(
                coherence=score_dict.get("coherence", 7.5),
                accuracy=score_dict.get("accuracy", 7.5),
                completeness=score_dict.get("completeness", 7.5),
                clarity=score_dict.get("clarity", 7.5),
                relevance=score_dict.get("relevance", 7.5),
                overall=score_dict.get("overall", 7.5),
                feedback=str(score_dict.get("feedback", "")),
                passed=True
            )
            aggregator.add_batch(samples, score)
            total_accepted = len(aggregator.samples)
            consecutive_failed = 0
        else:
            consecutive_failed += 1
            if batch.get("quality_score"):
                feedback = batch["quality_score"].get("feedback", "")
                errors.append(f"Batch failed quality check: {feedback[:50]}...")
            else:
                errors.append("Batch failed quality check with no score.")

        total_generated = state.get("total_generated", 0) + len(samples)

        # Check for HITL pause
        if config.ENABLE_HITL and total_accepted > 0:
            if total_accepted % config.HITL_CHECKPOINT_INTERVAL == 0:
                hitl_pause = True

        # Return only the updated keys
        return {
            "total_accepted": total_accepted,
            "total_generated": total_generated,
            "consecutive_failed_batches": consecutive_failed,
            "errors": errors,
            "current_step": "aggregated",
            "hitl_pause": hitl_pause
        }

    def should_continue(state) -> str:
        """Conditional edge: Check if we need more samples."""
        requirements = state.get("requirements", {})
        target = requirements.get("target_size", config.TARGET_SAMPLES)
        current = state.get("total_accepted", 0)
        consecutive_failures = state.get("consecutive_failed_batches", 0)

        if current >= target:
            return "export"
        elif state.get("hitl_pause", False):
            return "hitl"
        elif consecutive_failures >= config.MAX_CONSECUTIVE_FAILURES:
            return "export"
        else:
            return "continue"


    def hitl_checkpoint(state) -> Dict:
        """Node: Human-in-the-loop checkpoint"""
        total_accepted = state.get("total_accepted", 0)
        print(f"\n{'='*50}")
        print(f"HITL CHECKPOINT - {total_accepted} samples generated")
        print(f"{'='*50}")
        print("   Review current progress and quality scores.")
        print("   Dataset generation will continue automatically.")
        return {
            "hitl_pause": False,
            "current_step": "hitl_complete"
        }

    def stop_on_failure(state) -> Dict:
        """Node: Handles stopping due to too many consecutive failures."""
        consecutive_failed = state.get('consecutive_failed_batches', 0)
        print(f"\nStopping generation due to {consecutive_failed} consecutive failed batches.")
        return {
            "final_output": "Generation stopped due to excessive failures.",
            "current_step": "stopped_due_to_failure"
        }

    def export_dataset(state) -> Dict:
        """Node: Export final dataset"""
        print("\nExporting dataset...")

        df = aggregator.to_dataframe()
        stats = aggregator.get_stats()

        csv_path = exporter.export_csv(df, config.CSV_FILENAME)
        meta_path = exporter.export_metadata(stats)

        print(f"   CSV saved: {csv_path}")
        print(f"   Metadata saved: {meta_path}")
        print(f"   Total samples: {len(df)}")

        return {
            "final_output": csv_path,
            "current_step": "exported"
        }


    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("parse", parse_requirements)
    workflow.add_node("context", build_context)
    workflow.add_node("generate", generate_batch)
    workflow.add_node("quality", check_quality)
    workflow.add_node("aggregate", aggregate_results)
    workflow.add_node("hitl", hitl_checkpoint)
    workflow.add_node("stop_on_failure", stop_on_failure)
    workflow.add_node("export", export_dataset)

    # Add edges
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "context")
    workflow.add_edge("context", "generate")
    workflow.add_edge("generate", "quality")

    # Quality check loop
    workflow.add_conditional_edges(
        "quality",
        should_retry,
        {
            "aggregate": "aggregate",
            "retry": "context",
            "stop_generation": "stop_on_failure"
        }
    )

    # Continue or finish loop
    workflow.add_conditional_edges(
        "aggregate",
        should_continue,
        {
            "continue": "context",
            "hitl": "hitl",
            "export": "export"
        }
    )

    workflow.add_edge("hitl", "context")
    workflow.add_edge("stop_on_failure", END)
    workflow.add_edge("export", END)

    # Compile workflow (no checkpointer needed)
    app = workflow.compile()

    return app, aggregator


print("LangGraph workflow defined (FIXED VERSION)")
print("   Nodes: parse -> context -> generate -> quality -> aggregate -> export")
print("   Features: quality loop, HITL checkpoints, consecutive failure handling")
