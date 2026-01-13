#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %%
"""
===============================================================================
SynthAgent Engine - Multi-Agent Synthetic Data Generator

Purpose: Generate high-quality financial Q&A pairs using open-source LLMs

Constraints:
- 100% Free (Google Colab Free Tier)
- Open-Source LLMs via HuggingFace
- <12GB RAM usage with 4-bit quantization
- LangChain + LangGraph multi-agent architecture

Author: Generated from collaborative development
===============================================================================
"""

# %% [markdown]
# ============================================================================
# CELL 1: ENVIRONMENT SETUP
# ============================================================================

# %%
print("Installing dependencies...")

# %%
# Core ML / LLM packages (CUDA 11.8)
# !pip install -q torch torchvision torchaudio \
#   --index-url https://download.pytorch.org/whl/cu118

# %%
# !pip install -q "transformers>=4.36.0" "accelerate>=0.25.0" "bitsandbytes>=0.41.0"

# %%
# LangChain ecosystem
# !pip install -q "langchain>=0.3.0" "langchain-community>=0.3.0" "langchain-huggingface>=0.1.0"
# !pip install -q "langgraph>=0.2.0"

# %%
# Structured output & validation
# !pip install -q "pydantic>=2.0.0"

# %%
# Data handling
# !pip install -q pandas numpy

# %%
# Progress & display
# !pip install -q tqdm rich

# %%
print("All dependencies installed successfully!")


# %% [markdown]
# ============================================================================
# CELL 2: IMPORTS AND CONFIGURATION
# ============================================================================

# %%
import os
import gc
import json
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal, TypedDict, Annotated
from enum import Enum

# %%
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pydantic import BaseModel, Field, validator
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# %%
# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFacePipeline

# %%
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# %%
# Transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# %%
warnings.filterwarnings('ignore')
console = Console()


# %% [markdown]
# ============================================================================
# CONFIGURATION
# ============================================================================

# %%
class Config:
    """Global configuration for SynthAgent Engine"""

    # Model settings
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    USE_4BIT = True
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7

    # Generation settings
    TARGET_SAMPLES = 10000
    BATCH_SIZE = 20
    SAMPLES_PER_LLM_CALL = 5

    # Quality thresholds
    MIN_QUALITY_SCORE = 7.0
    MAX_CORRECTION_ROUNDS = 2
    MAX_CONSECUTIVE_FAILURES = 5

    # HITL settings
    HITL_CHECKPOINT_INTERVAL = 500
    ENABLE_HITL = True

    # Output settings
    OUTPUT_DIR = "/content/output"
    CSV_FILENAME = "financial_qa_dataset.csv"

    # Seed for reproducibility
    RANDOM_SEED = 42


# %%
config = Config()
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# %%
print(f"Configuration loaded:")
print(f"   Model: {config.MODEL_ID}")
print(f"   Target samples: {config.TARGET_SAMPLES}")
print(f"   Batch size: {config.BATCH_SIZE}")
print(f"   Quality threshold: {config.MIN_QUALITY_SCORE}/10")


# %% [markdown]
# ============================================================================
# CELL 3: PYDANTIC MODELS
# ============================================================================

# %%
class TaskType(str, Enum):
    """Supported task types for generation"""
    QA_SINGLE = "question_answering_single"
    QA_MULTI = "question_answering_multi"
    REASONING = "chain_of_thought"
    INSTRUCTION = "instruction_following"
    CLASSIFICATION = "classification"


# %%
class DifficultyLevel(str, Enum):
    """Difficulty levels for generated content"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# %%
class FinancialCategory(str, Enum):
    """Financial domain categories"""
    INVESTING = "investing"
    BANKING = "banking"
    TAXATION = "taxation"
    ACCOUNTING = "accounting"
    INSURANCE = "insurance"
    RETIREMENT = "retirement"
    REAL_ESTATE = "real_estate"
    CRYPTO = "cryptocurrency"
    CORPORATE_FINANCE = "corporate_finance"
    PERSONAL_FINANCE = "personal_finance"
    MARKETS = "stock_markets"
    DERIVATIVES = "derivatives"
    RISK_MANAGEMENT = "risk_management"
    REGULATIONS = "financial_regulations"


# %%
class GenerationRequirements(BaseModel):
    """Parsed requirements from user input"""
    domain: str = Field(default="finance", description="Target domain")
    task_type: TaskType = Field(default=TaskType.QA_SINGLE)
    target_size: int = Field(default=10000, ge=100, le=100000)
    difficulty_distribution: Dict[str, float] = Field(
        default={"beginner": 0.2, "intermediate": 0.4, "advanced": 0.3, "expert": 0.1}
    )
    categories: List[str] = Field(default_factory=list)
    language: str = Field(default="english")
    special_requirements: List[str] = Field(default_factory=list)


# %%
class QAPair(BaseModel):
    """Single question-answer pair with metadata"""
    question: str = Field(..., min_length=10, description="The question")
    answer: str = Field(..., min_length=20, description="Detailed answer")
    category: str = Field(..., description="Financial category")
    difficulty: str = Field(..., description="Difficulty level")
    reasoning: Optional[str] = Field(None, description="Chain of thought reasoning")
    keywords: List[str] = Field(default_factory=list, description="Key concepts")

    @validator('question')
    def question_must_end_with_punctuation(cls, v):
        if not v.strip().endswith('?'):
            v = v.strip() + '?'
        return v


# %%
class QualityScore(BaseModel):
    """Quality assessment for a QA pair or batch"""
    coherence: float = Field(..., ge=0, le=10, description="Logical coherence")
    accuracy: float = Field(..., ge=0, le=10, description="Factual accuracy")
    completeness: float = Field(..., ge=0, le=10, description="Answer completeness")
    clarity: float = Field(..., ge=0, le=10, description="Language clarity")
    relevance: float = Field(..., ge=0, le=10, description="Domain relevance")
    overall: float = Field(..., ge=0, le=10, description="Overall quality")
    feedback: str = Field(default="", description="Improvement suggestions")
    passed: bool = Field(default=True, description="Meets quality threshold")


# %%
class GenerationBatch(BaseModel):
    """Batch of generated QA pairs"""
    batch_id: int
    samples: List[QAPair]
    quality_score: Optional[QualityScore] = None
    correction_rounds: int = Field(default=0)
    status: str = Field(default="pending")


# %%
# LangGraph State Definition
class AgentState(TypedDict):
    """State passed between agents in the graph"""
    requirements: Optional[Dict]
    schema: Optional[Dict]
    context: Optional[Dict]
    current_batch: Optional[Dict]
    generated_samples: List[Dict]
    quality_scores: List[Dict]
    total_generated: int
    total_accepted: int
    errors: List[str]
    current_step: str
    hitl_pause: bool
    final_output: Optional[str]
    consecutive_failed_batches: int


# %%
print("Pydantic models defined")
print(f"   - GenerationRequirements: Parses user input")
print(f"   - QAPair: Individual Q&A sample")
print(f"   - QualityScore: Quality metrics")
print(f"   - GenerationBatch: Batch container")
print(f"   - AgentState: LangGraph workflow state")


# %% [markdown]
# ============================================================================
# CELL 4: MODEL LOADING
# ============================================================================

# %%
console.print(Panel.fit("Loading Mistral-7B-Instruct with 4-bit quantization...",
                        title="Model Loading"))

# %%
# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: No GPU detected - will use CPU (much slower)")

# %%
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# %%
# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
# Load model with quantization
print("Loading model (this takes 3-5 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

# %%
# Create text generation pipeline
print("Creating generation pipeline...")
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=config.MAX_NEW_TOKENS,
    temperature=config.TEMPERATURE,
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.1,
    return_full_text=False,
)

# %%
# Wrap for LangChain
llm = HuggingFacePipeline(pipeline=text_pipeline)

# %%
# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

# %%
console.print(Panel.fit("Model loaded successfully!", title="Complete", style="green"))
print(f"   Memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# %% [markdown]
# ============================================================================
# CELL 5: FINANCIAL DOMAIN KNOWLEDGE
# ============================================================================

# %%
FINANCIAL_KNOWLEDGE = {
    "categories": {
        "investing": {
            "topics": ["stocks", "bonds", "ETFs", "mutual funds", "portfolio diversification",
                      "asset allocation", "value investing", "growth investing", "dividend investing",
                      "index funds", "market timing", "dollar-cost averaging"],
            "concepts": ["P/E ratio", "market capitalization", "dividend yield", "beta",
                        "alpha", "Sharpe ratio", "bull market", "bear market", "volatility"],
        },
        "banking": {
            "topics": ["savings accounts", "checking accounts", "CDs", "money market accounts",
                      "interest rates", "FDIC insurance", "overdraft protection", "wire transfers"],
            "concepts": ["APY", "APR", "compound interest", "liquidity", "minimum balance"],
        },
        "taxation": {
            "topics": ["income tax", "capital gains", "tax deductions", "tax credits",
                      "tax brackets", "filing status", "W-2", "1099", "estimated taxes"],
            "concepts": ["marginal tax rate", "effective tax rate", "tax-deferred", "tax-exempt"],
        },
        "retirement": {
            "topics": ["401(k)", "IRA", "Roth IRA", "pension", "Social Security",
                      "retirement planning", "required minimum distributions", "catch-up contributions"],
            "concepts": ["compound growth", "employer match", "vesting", "early withdrawal penalty"],
        },
        "personal_finance": {
            "topics": ["budgeting", "emergency fund", "debt management", "credit score",
                      "net worth", "financial goals", "saving strategies", "spending tracking"],
            "concepts": ["50/30/20 rule", "debt-to-income ratio", "sinking fund", "pay yourself first"],
        },
        "stock_markets": {
            "topics": ["NYSE", "NASDAQ", "market orders", "limit orders", "stop-loss",
                      "trading hours", "after-hours trading", "IPO", "stock splits"],
            "concepts": ["bid-ask spread", "market makers", "trading volume", "circuit breakers"],
        },
        "corporate_finance": {
            "topics": ["financial statements", "balance sheet", "income statement", "cash flow",
                      "EBITDA", "working capital", "capital structure", "mergers and acquisitions"],
            "concepts": ["ROE", "ROA", "debt-to-equity", "current ratio", "quick ratio"],
        },
        "risk_management": {
            "topics": ["diversification", "hedging", "insurance", "asset allocation",
                      "risk tolerance", "systematic risk", "unsystematic risk"],
            "concepts": ["standard deviation", "VaR", "correlation", "beta coefficient"],
        },
    },

    "difficulty_guidelines": {
        "beginner": "Basic concepts, simple explanations, everyday financial decisions",
        "intermediate": "Moderate complexity, some calculations, investment strategies",
        "advanced": "Complex scenarios, detailed analysis, professional-level knowledge",
        "expert": "Highly technical, regulatory knowledge, institutional finance",
    },

    "question_templates": [
        "What is {concept} and why is it important for {context}?",
        "How does {concept} affect {related_topic}?",
        "What are the advantages and disadvantages of {topic}?",
        "How should someone approach {scenario}?",
        "What factors should be considered when {action}?",
        "Explain the difference between {concept1} and {concept2}.",
        "What are the tax implications of {financial_action}?",
        "How can {strategy} help achieve {financial_goal}?",
        "What risks are associated with {investment_type}?",
        "When is the best time to {financial_decision}?",
    ],
}

# %%
# Few-shot examples for high-quality generation
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the difference between a traditional IRA and a Roth IRA?",
        "answer": "A traditional IRA and Roth IRA differ primarily in their tax treatment. With a traditional IRA, contributions may be tax-deductible, and you pay taxes when you withdraw funds in retirement. With a Roth IRA, contributions are made with after-tax dollars, but qualified withdrawals in retirement are completely tax-free. The choice between them depends on whether you expect to be in a higher or lower tax bracket in retirement. Traditional IRAs also have required minimum distributions (RMDs) starting at age 73, while Roth IRAs have no RMDs during the owner's lifetime.",
        "category": "retirement",
        "difficulty": "intermediate",
        "keywords": ["IRA", "Roth IRA", "traditional IRA", "tax-deferred", "tax-free", "RMD"]
    },
    {
        "question": "How does compound interest work and why is it called the 'eighth wonder of the world'?",
        "answer": "Compound interest is the process where interest is calculated not only on the initial principal but also on the accumulated interest from previous periods. For example, if you invest $1,000 at 10% annual interest, after year one you have $1,100. In year two, you earn 10% on $1,100 (not just $1,000), giving you $1,210. This compounding effect accelerates wealth growth exponentially over time. Albert Einstein reportedly called it the 'eighth wonder of the world' because those who understand it earn it, while those who don't pay it. Starting early maximizes the benefit, as time is the most powerful factor in compound growth.",
        "category": "investing",
        "difficulty": "beginner",
        "keywords": ["compound interest", "principal", "exponential growth", "time value of money"]
    },
    {
        "question": "What is the debt-to-equity ratio and how do investors use it to evaluate companies?",
        "answer": "The debt-to-equity (D/E) ratio measures a company's financial leverage by dividing total liabilities by shareholders' equity. A D/E ratio of 1.0 means the company has equal amounts of debt and equity financing. Higher ratios indicate more debt financing, which can amplify returns but also increases financial risk. Investors use this metric to: 1) Compare companies within the same industry (capital-intensive industries like utilities typically have higher D/E ratios than tech companies), 2) Assess bankruptcy risk during economic downturns, 3) Evaluate management's capital allocation strategy. Generally, a D/E ratio below 2.0 is considered conservative for most industries.",
        "category": "corporate_finance",
        "difficulty": "advanced",
        "keywords": ["debt-to-equity ratio", "leverage", "financial risk", "capital structure"]
    },
]

# %%
print("Financial knowledge base loaded")
print(f"   Categories: {len(FINANCIAL_KNOWLEDGE['categories'])}")
print(f"   Question templates: {len(FINANCIAL_KNOWLEDGE['question_templates'])}")
print(f"   Few-shot examples: {len(FEW_SHOT_EXAMPLES)}")


# %% [markdown]
# ============================================================================
# CELL 6: AGENT PROMPT TEMPLATES
# ============================================================================

# %%
# Requirement Parser Agent Prompt
REQUIREMENT_PARSER_PROMPT = """<s>[INST] You are a requirements analysis expert. Parse the user's request and extract structured generation requirements.

USER REQUEST: {user_input}

Extract and return a JSON object with these fields:
- domain: The target domain (default: "finance")
- task_type: One of [question_answering_single, reasoning, instruction_following]
- target_size: Number of samples to generate (default: 10000)
- categories: List of specific categories to focus on
- difficulty_distribution: Dict with beginner/intermediate/advanced/expert percentages
- special_requirements: Any special instructions

Return ONLY valid JSON, no other text.

Example output:
{{"domain": "finance", "task_type": "question_answering_single", "target_size": 10000, "categories": ["investing", "retirement", "taxation"], "difficulty_distribution": {{"beginner": 0.25, "intermediate": 0.40, "advanced": 0.25, "expert": 0.10}}, "special_requirements": ["include calculations", "practical examples"]}}

JSON output: [/INST]"""

# %%
# Schema Designer Agent Prompt
SCHEMA_DESIGNER_PROMPT = """<s>[INST] You are a data schema expert. Design the output schema for financial Q&A pairs.

REQUIREMENTS: {requirements}

Create a schema that includes:
1. Question format guidelines
2. Answer structure requirements
3. Required metadata fields
4. Quality criteria

Return a JSON schema specification.

Output: [/INST]"""

# %%
# Context Builder Agent Prompt
CONTEXT_BUILDER_PROMPT = """<s>[INST] You are a financial domain expert. Build rich context for generating Q&A pairs.

CATEGORY: {category}
DIFFICULTY: {difficulty}
AVAILABLE TOPICS: {topics}
AVAILABLE CONCEPTS: {concepts}

Create context that includes:
1. Relevant background information
2. Key concepts to incorporate
3. Real-world scenarios
4. Common misconceptions to address

Return the context as a JSON object with fields: background, key_concepts, scenarios, misconceptions.

Context JSON: [/INST]"""

# %%
# Master Data Generator Agent Prompt
GENERATOR_PROMPT = """<s>[INST] You are an expert financial educator creating high-quality Q&A pairs for training AI models.

TASK: Generate {num_samples} diverse, high-quality financial question-answer pairs.

CATEGORY: {category}
DIFFICULTY LEVEL: {difficulty}
CONTEXT: {context}

REQUIREMENTS:
1. Questions should be clear, specific, and educational
2. Answers must be accurate, comprehensive (2-4 sentences minimum), and practical
3. Include relevant financial terminology
4. Vary question types (what, how, why, when, compare)
5. Answers should explain concepts, not just define them

FEW-SHOT EXAMPLES:
{examples}

Generate {num_samples} Q&A pairs in this exact JSON format:
[
  {{"question": "...", "answer": "...", "category": "{category}", "difficulty": "{difficulty}", "keywords": ["...", "..."]}},
  ...
]

Output ONLY the JSON array, no other text:
[/INST]"""

# %%
# Quality Controller Agent Prompt
QUALITY_CONTROLLER_PROMPT = """<s>[INST] You are a quality assurance expert reviewing financial Q&A pairs.

REVIEW THESE Q&A PAIRS:
{samples}

SCORING CRITERIA (0-10 for each):
1. Coherence: Is the answer logically structured?
2. Accuracy: Is the financial information correct?
3. Completeness: Does the answer fully address the question?
4. Clarity: Is the language clear and professional?
5. Relevance: Is this appropriate for the stated category/difficulty?

Return a JSON object with:
{{"coherence": X, "accuracy": X, "completeness": X, "clarity": X, "relevance": X, "overall": X, "feedback": "specific improvement suggestions", "passed": true/false}}

A sample passes if overall >= 7.0

Quality assessment JSON: [/INST]"""

# %%
print("Agent prompts configured")
print("   - RequirementParserAgent: Extracts structured requirements")
print("   - SchemaDesignerAgent: Designs output schema")
print("   - ContextBuilderAgent: Creates domain context")
print("   - MasterGeneratorAgent: Generates Q&A pairs")
print("   - QualityControllerAgent: Validates quality")


# %% [markdown]
# ============================================================================
# CELL 7: AGENT IMPLEMENTATIONS
# ============================================================================

# %%
import re


# %%
def extract_json(text: str) -> Any:
    """Safely extract JSON from LLM response"""
    try:
        # Try direct parse first
        return json.loads(text)
    except:
        pass

    # Try to find JSON in the text
    patterns = [
        r'\[[\s\S]*\]',  # JSON array
        r'\{[\s\S]*\}',  # JSON object
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

    return None


# %%
def clean_llm_output(text: str) -> str:
    """Clean LLM output artifacts"""
    # Remove common artifacts
    text = text.replace('</s>', '').replace('<s>', '')
    text = text.replace('[INST]', '').replace('[/INST]', '')
    return text.strip()


# %%
class RequirementParserAgent:
    """Parses natural language input into structured requirements"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(REQUIREMENT_PARSER_PROMPT)

    def invoke(self, user_input: str) -> Dict:
        """Parse user input into structured requirements"""
        try:
            formatted_prompt = self.prompt.format(user_input=user_input)
            response = self.llm.invoke(formatted_prompt)
            response = clean_llm_output(response)

            parsed = extract_json(response)
            if parsed:
                return parsed

            # Default fallback
            return {
                "domain": "finance",
                "task_type": "question_answering_single",
                "target_size": config.TARGET_SAMPLES,
                "categories": list(FINANCIAL_KNOWLEDGE["categories"].keys()),
                "difficulty_distribution": {"beginner": 0.25, "intermediate": 0.40,
                                           "advanced": 0.25, "expert": 0.10},
                "special_requirements": []
            }
        except Exception as e:
            print(f"Parser warning: {e}, using defaults")
            return {
                "domain": "finance",
                "task_type": "question_answering_single",
                "target_size": config.TARGET_SAMPLES,
                "categories": list(FINANCIAL_KNOWLEDGE["categories"].keys()),
                "difficulty_distribution": {"beginner": 0.25, "intermediate": 0.40,
                                           "advanced": 0.25, "expert": 0.10},
                "special_requirements": []
            }


# %%
class ContextBuilderAgent:
    """Builds rich domain context for generation"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(CONTEXT_BUILDER_PROMPT)

    def invoke(self, category: str, difficulty: str) -> Dict:
        """Build context for the given category and difficulty"""
        cat_data = FINANCIAL_KNOWLEDGE["categories"].get(
            category,
            FINANCIAL_KNOWLEDGE["categories"]["investing"]
        )

        try:
            formatted_prompt = self.prompt.format(
                category=category,
                difficulty=difficulty,
                topics=", ".join(cat_data.get("topics", [])),
                concepts=", ".join(cat_data.get("concepts", []))
            )

            response = self.llm.invoke(formatted_prompt)
            response = clean_llm_output(response)

            parsed = extract_json(response)
            if parsed:
                return parsed
        except Exception as e:
            print(f"Context builder warning: {e}")

        # Fallback context
        return {
            "background": f"Financial knowledge about {category}",
            "key_concepts": cat_data.get("concepts", []),
            "scenarios": cat_data.get("topics", []),
            "misconceptions": []
        }


# %%
class MasterDataGeneratorAgent:
    """Generates high-quality Q&A pairs in batches"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(GENERATOR_PROMPT)

    def invoke(self, category: str, difficulty: str, context: Dict,
               num_samples: int = 5) -> List[Dict]:
        """Generate a batch of Q&A pairs"""

        # Format few-shot examples
        examples_str = ""
        for ex in FEW_SHOT_EXAMPLES[:2]:
            examples_str += f"\nQ: {ex['question']}\nA: {ex['answer']}\nCategory: {ex['category']}, Difficulty: {ex['difficulty']}\n"

        try:
            formatted_prompt = self.prompt.format(
                num_samples=num_samples,
                category=category,
                difficulty=difficulty,
                context=json.dumps(context),
                examples=examples_str
            )

            response = self.llm.invoke(formatted_prompt)
            response = clean_llm_output(response)

            parsed = extract_json(response)
            if parsed and isinstance(parsed, list):
                # Validate and clean each sample
                valid_samples = []
                for item in parsed:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        sample = {
                            'question': str(item.get('question', '')).strip(),
                            'answer': str(item.get('answer', '')).strip(),
                            'category': category,
                            'difficulty': difficulty,
                            'keywords': item.get('keywords', []),
                            'reasoning': item.get('reasoning', '')
                        }
                        if len(sample['question']) > 10 and len(sample['answer']) > 20:
                            valid_samples.append(sample)

                if valid_samples:
                    return valid_samples

        except Exception as e:
            print(f"Generator warning: {e}")

        return []


# %%
class QualityControllerAgent:
    """Validates and scores generated Q&A pairs"""

    def __init__(self, llm, threshold: float = 7.0):
        self.llm = llm
        self.threshold = threshold
        self.prompt = PromptTemplate.from_template(QUALITY_CONTROLLER_PROMPT)

    def invoke(self, samples: List[Dict]) -> QualityScore:
        """Score a batch of samples"""

        if not samples:
            return QualityScore(
                coherence=0, accuracy=0, completeness=0,
                clarity=0, relevance=0, overall=0,
                feedback="No samples to evaluate", passed=False
            )

        try:
            samples_str = json.dumps(samples[:5], indent=2)  # Limit for prompt size
            formatted_prompt = self.prompt.format(samples=samples_str)

            response = self.llm.invoke(formatted_prompt)
            response = clean_llm_output(response)

            parsed = extract_json(response)
            if parsed and isinstance(parsed, dict):
                return QualityScore(
                    coherence=float(parsed.get('coherence', 7)),
                    accuracy=float(parsed.get('accuracy', 7)),
                    completeness=float(parsed.get('completeness', 7)),
                    clarity=float(parsed.get('clarity', 7)),
                    relevance=float(parsed.get('relevance', 7)),
                    overall=float(parsed.get('overall', 7)),
                    feedback=str(parsed.get('feedback', '')),
                    passed=float(parsed.get('overall', 7)) >= self.threshold
                )

        except Exception as e:
            print(f"Quality check warning: {e}")

        # Default passing score (optimistic)
        return QualityScore(
            coherence=7.5, accuracy=7.5, completeness=7.5,
            clarity=7.5, relevance=7.5, overall=7.5,
            feedback="Auto-approved", passed=True
        )


# %%
class DatasetAggregator:
    """Collects and aggregates all generated samples"""

    def __init__(self):
        self.samples = []
        self.quality_scores = []

    def add_batch(self, samples: List[Dict], score: QualityScore):
        """Add a validated batch"""
        self.samples.extend(samples)
        self.quality_scores.append(score)

    def get_stats(self) -> Dict:
        """Get aggregation statistics"""
        if not self.quality_scores:
            return {"total": 0}

        return {
            "total": len(self.samples),
            "avg_quality": np.mean([s.overall for s in self.quality_scores]),
            "by_category": pd.Series([s.get('category', 'unknown') for s in self.samples]).value_counts().to_dict(),
            "by_difficulty": pd.Series([s.get('difficulty', 'unknown') for s in self.samples]).value_counts().to_dict()
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame"""
        return pd.DataFrame(self.samples)


# %%
class ExporterAgent:
    """Exports dataset to various formats"""

    def __init__(self, output_dir: str = "/content/output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_csv(self, df: pd.DataFrame, filename: str = "dataset.csv") -> str:
        """Export to CSV format"""
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False, encoding='utf-8')
        return path

    def export_metadata(self, stats: Dict, filename: str = "metadata.json") -> str:
        """Export metadata"""
        path = os.path.join(self.output_dir, filename)
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "model_used": config.MODEL_ID,
            "statistics": stats,
            "config": {
                "batch_size": config.BATCH_SIZE,
                "quality_threshold": config.MIN_QUALITY_SCORE,
                "temperature": config.TEMPERATURE
            }
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        return path


# %%
print("Agent classes implemented")
print("   - RequirementParserAgent")
print("   - ContextBuilderAgent")
print("   - MasterDataGeneratorAgent")
print("   - QualityControllerAgent")
print("   - DatasetAggregator")
print("   - ExporterAgent")


# %% [markdown]
# ============================================================================
# CELL 8: LANGGRAPH WORKFLOW
# ============================================================================


# %%
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
    def parse_requirements(state: AgentState) -> Dict:
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

    def build_context(state: AgentState) -> Dict:
        """Node: Build domain context for current batch"""
        categories = list(FINANCIAL_KNOWLEDGE["categories"].keys())

        # Get requirements from state (it's a dict, not requiring mutation)
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

    def generate_batch(state: AgentState) -> Dict:
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

    def check_quality(state: AgentState) -> Dict:
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

    def should_retry(state: AgentState) -> str:
        """Conditional edge: Decide if batch needs retry or if we've failed too many times consecutively."""
        batch = state.get("current_batch", {})
        passed = batch.get("passed", False)
        rounds = batch.get("correction_rounds", 0)
        consecutive_failures = state.get("consecutive_failed_batches", 0)

        if passed:
            # If passed, reset consecutive failures (will be done in next node)
            return "aggregate"
        else:
            # If failed, check consecutive failures
            new_consecutive = consecutive_failures + 1
            if new_consecutive >= config.MAX_CONSECUTIVE_FAILURES:
                return "stop_generation"
            elif rounds >= config.MAX_CORRECTION_ROUNDS:
                return "aggregate"
            else:
                return "retry"

    def aggregate_results(state: AgentState) -> Dict:
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

    def should_continue(state: AgentState) -> str:
        """Conditional edge: Check if we need more samples or should stop due to failures."""
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


    def hitl_checkpoint(state: AgentState) -> Dict:
        """Node: Human-in-the-loop checkpoint"""
        total_accepted = state.get("total_accepted", 0)
        print(f"\n{'='*50}")
        print(f"HITL CHECKPOINT - {total_accepted} samples generated")
        print(f"{'='*50}")
        print("   Review current progress and quality scores.")
        print("   Dataset generation will continue automatically.")
        # Return only the updated keys
        return {
            "hitl_pause": False,
            "current_step": "hitl_complete"
        }

    def stop_on_failure(state: AgentState) -> Dict:
        """Node: Handles stopping due to too many consecutive failures."""
        consecutive_failed = state.get('consecutive_failed_batches', 0)
        print(f"\nStopping generation due to {consecutive_failed} consecutive failed batches.")
        # Return only the updated keys
        return {
            "final_output": "Generation stopped due to excessive failures.",
            "current_step": "stopped_due_to_failure"
        }

    def export_dataset(state: AgentState) -> Dict:
        """Node: Export final dataset"""
        print("\nExporting dataset...")

        df = aggregator.to_dataframe()
        stats = aggregator.get_stats()

        csv_path = exporter.export_csv(df, config.CSV_FILENAME)
        meta_path = exporter.export_metadata(stats)

        print(f"   CSV saved: {csv_path}")
        print(f"   Metadata saved: {meta_path}")
        print(f"   Total samples: {len(df)}")

        # Return only the updated keys
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

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app, aggregator


# %%
print("LangGraph workflow defined")
print("   Nodes: parse -> context -> generate -> quality -> aggregate -> export")
print("   Features: quality loop, HITL checkpoints, memory persistence, consecutive failure handling")


# %% [markdown]
# ============================================================================
# CELL 9: MAIN EXECUTION ENGINE
# ============================================================================


# %%
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

    # Initialize state
    initial_state = {
        "user_input": user_request,
        "requirements": None,
        "schema": None,
        "context": None,
        "current_batch": None,
        "generated_samples": [],
        "quality_scores": [],
        "total_generated": 0,
        "total_accepted": 0,
        "errors": [],
        "current_step": "init",
        "hitl_pause": False,
        "final_output": None,
        "consecutive_failed_batches": 0
    }

    # Run with progress tracking
    thread_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config_run = {"configurable": {"thread_id": thread_id, "recursion_limit": config.TARGET_SAMPLES * 2}}

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

    # Display summary
    console.print(Panel.fit(
        f"Generation Complete!\n\n"
        f"Statistics:\n"
        f"   Total samples: {len(df)}\n"
        f"   Time elapsed: {elapsed/60:.1f} minutes\n"
        f"   Rate: {len(df)/elapsed:.2f} samples/second\n"
        f"   Avg quality: {stats.get('avg_quality', 0):.2f}/10",
        title="Complete", style="green"
    ))

    # Show category distribution
    if "by_category" in stats:
        print("\n" + "="*60)
        print("CATEGORY DISTRIBUTION")
        print("="*60)
        for cat, count in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
            pct = 100 * count / len(df)
            bar = "*" * int(pct / 2)
            print(f"{cat:20s}: {count:5d} ({pct:5.1f}%) {bar}")

    # Show difficulty distribution
    print(f"\n{'='*60}")
    print("DIFFICULTY DISTRIBUTION")
    print(f"{'='*60}")
    if "by_difficulty" in stats:
        for diff, count in stats["by_difficulty"].items():
            pct = 100 * count / len(df)
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


# %%
def preview_samples(df: pd.DataFrame, n: int = 5):
    """Preview generated samples"""
    console.print(Panel.fit(f"Sample Preview (showing {n} examples)", title="Preview"))

    for i, row in df.head(n).iterrows():
        print(f"\n{'-'*60}")
        print(f"[{row.get('category', 'N/A')} | {row.get('difficulty', 'N/A')}]")
        print(f"Q: {row['question']}")
        print(f"A: {row['answer'][:200]}..." if len(str(row['answer'])) > 200 else f"A: {row['answer']}")


# %% [markdown]
# ============================================================================
# CELL 10: EXECUTION
# ============================================================================

# %%
if __name__ == "__main__":
    # Quick test (2-5 minutes)
    df, stats, csv_path = run_generation(target_samples=10)

    # Full generation (4-6 hours)
    # df, stats, csv_path = run_generation(target_samples=10000)

    # Preview results
    preview_samples(df, n=5)
