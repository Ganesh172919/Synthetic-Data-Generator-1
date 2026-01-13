"""
FIXED IMPORTS AND STATE DEFINITION - Copy this to replace your notebook cells

This fixes the LangGraph state type issues.
"""

# ============================================================================
# CELL 2: IMPORTS AND CONFIGURATION (FIXED)
# ============================================================================

import os
import gc
import json
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, TypedDict, Annotated
from enum import Enum

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFacePipeline

# LangGraph imports (FIXED - no MemorySaver needed)
from langgraph.graph import StateGraph, END

# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

warnings.filterwarnings('ignore')
console = Console()


# ============================================================================
# PYDANTIC MODELS (FIXED AgentState)
# ============================================================================

# LangGraph State Definition (FIXED - use Dict not Optional[Dict])
class AgentState(TypedDict):
    """State passed between agents in the graph"""
    user_input: str
    requirements: Dict
    schema: Dict
    context: Dict
    current_batch: Dict
    generated_samples: List[Dict]
    quality_scores: List[Dict]
    total_generated: int
    total_accepted: int
    errors: List[str]
    current_step: str
    hitl_pause: bool
    final_output: str
    consecutive_failed_batches: int


print("Imports loaded and AgentState defined (FIXED VERSION)")
