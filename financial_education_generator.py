"""
Financial Education Dataset Generator
=====================================
A multi-agent system for generating 20k high-quality educational financial Q&A pairs.
Optimized for Google Colab with Hugging Face models and 4-bit quantization.

Features:
- Uses Mistral-7B-Instruct with 4-bit quantization (fits in Colab free tier)
- Generates data in segments to avoid memory issues
- Saves progress continuously to prevent data loss
- Includes deduplication and quality checks
- Outputs JSONL format ready for training

Usage in Google Colab:
1. Run all cells or execute: !python financial_education_generator.py
2. Dataset will be saved to 'financial_education_dataset.jsonl'
"""

# ============================================================================
# SECTION 1: INSTALLATION AND IMPORTS
# ============================================================================

import subprocess
import sys

def install_dependencies():
    """Install required packages for Google Colab."""
    packages = [
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "tqdm",
        "jsonlines"
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except subprocess.CalledProcessError:
            print(f"⚠️ Warning: Could not install {package}")
    print("✅ Dependencies installed successfully!\n")

# Run installation
install_dependencies()

# Now import all required libraries
import os
import json
import time
import random
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import warnings

# Suppress all transformers and torch warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
warnings.filterwarnings("ignore", message=".*sequential.*GPU.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable to suppress pipeline warning
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging as transformers_logging
)

# Suppress transformers logging
transformers_logging.set_verbosity_error()

from tqdm import tqdm

# Try importing sentence-transformers for deduplication
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️ sentence-transformers not available. Using hash-based deduplication.")

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the dataset generator."""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_quantization: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Dataset settings
    target_dataset_size: int = 10  # Changed to 10 for testing
    batch_size: int = 5  # Questions per batch
    save_interval: int = 5  # Save every N Q&A pairs (adjusted for small dataset)
    
    # Output settings
    output_file: str = "financial_education_dataset.jsonl"
    checkpoint_file: str = "generator_checkpoint.json"
    log_file: str = "generator.log"
    
    # Quality settings
    min_answer_length: int = 100
    max_retries: int = 3
    similarity_threshold: float = 0.85  # For deduplication
    
    # Memory management
    clear_cache_interval: int = 100  # Clear GPU cache every N generations


# Initialize config
CONFIG = Config()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 3: FINANCIAL CURRICULUM STRUCTURE
# ============================================================================

FINANCIAL_CURRICULUM = {
    "Personal Finance": {
        "subtopics": [
            "Budgeting Basics", "Emergency Funds", "Debt Management",
            "Credit Scores", "Saving Strategies", "Insurance Fundamentals",
            "Retirement Planning", "Tax Basics", "Financial Goal Setting",
            "Net Worth Calculation"
        ],
        "difficulty_range": ["beginner", "intermediate"]
    },
    "Investing Fundamentals": {
        "subtopics": [
            "Stock Market Basics", "Bonds and Fixed Income", "Mutual Funds",
            "ETFs", "Index Investing", "Diversification", "Risk vs Return",
            "Dollar Cost Averaging", "Dividend Investing", "Value Investing"
        ],
        "difficulty_range": ["beginner", "intermediate", "advanced"]
    },
    "Advanced Investing": {
        "subtopics": [
            "Options Trading", "Futures and Derivatives", "Technical Analysis",
            "Fundamental Analysis", "Portfolio Management", "Asset Allocation",
            "Hedge Funds", "Private Equity", "Real Estate Investment",
            "Alternative Investments"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Macroeconomics": {
        "subtopics": [
            "GDP and Economic Growth", "Inflation and Deflation",
            "Interest Rates", "Monetary Policy", "Fiscal Policy",
            "Business Cycles", "Unemployment", "International Trade",
            "Exchange Rates", "Central Banking"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Corporate Finance": {
        "subtopics": [
            "Financial Statements", "Ratio Analysis", "Cash Flow Management",
            "Capital Budgeting", "Cost of Capital", "Capital Structure",
            "Mergers and Acquisitions", "IPOs", "Corporate Valuation",
            "Working Capital Management"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Cryptocurrency and Blockchain": {
        "subtopics": [
            "Bitcoin Basics", "Blockchain Technology", "Altcoins",
            "Crypto Wallets", "Decentralized Finance (DeFi)",
            "NFTs", "Crypto Exchanges", "Mining", "Smart Contracts",
            "Crypto Regulation"
        ],
        "difficulty_range": ["beginner", "intermediate", "advanced"]
    },
    "Behavioral Finance": {
        "subtopics": [
            "Cognitive Biases", "Emotional Investing", "Herd Mentality",
            "Loss Aversion", "Overconfidence Bias", "Anchoring",
            "Mental Accounting", "Market Psychology", "Investor Behavior",
            "Decision Making Under Uncertainty"
        ],
        "difficulty_range": ["intermediate", "advanced"]
    },
    "Financial Markets": {
        "subtopics": [
            "Stock Exchanges", "Bond Markets", "Forex Markets",
            "Commodity Markets", "Money Markets", "Market Participants",
            "Market Efficiency", "Market Regulations", "Trading Mechanisms",
            "Market Indices"
        ],
        "difficulty_range": ["beginner", "intermediate", "advanced"]
    }
}

QUESTION_TYPES = [
    "definition",      # "What is X?"
    "conceptual",      # "How does X work?"
    "comparison",      # "What is the difference between X and Y?"
    "example",         # "Can you give an example of X?"
    "application",     # "How can I apply X in practice?"
    "misconception",   # "What is a common misconception about X?"
    "importance",      # "Why is X important?"
    "calculation",     # "How do you calculate X?"
    "strategy",        # "What strategies can be used for X?"
    "risk"            # "What are the risks associated with X?"
]

# ============================================================================
# SECTION 4: DATA STRUCTURES
# ============================================================================

@dataclass
class QAPair:
    """Represents a Question-Answer pair."""
    id: str
    topic: str
    subtopic: str
    question: str
    answer: str
    difficulty: str
    question_type: str
    review_status: str = "pending"
    sources: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class GenerationState:
    """Tracks the current state of generation."""
    total_generated: int = 0
    total_verified: int = 0
    total_rejected: int = 0
    current_topic_idx: int = 0
    current_subtopic_idx: int = 0
    question_hashes: set = field(default_factory=set)
    answer_hashes: set = field(default_factory=set)
    
    def save_checkpoint(self, filepath: str):
        """Save current state to checkpoint file."""
        state_dict = {
            "total_generated": self.total_generated,
            "total_verified": self.total_verified,
            "total_rejected": self.total_rejected,
            "current_topic_idx": self.current_topic_idx,
            "current_subtopic_idx": self.current_subtopic_idx,
            "question_hashes": list(self.question_hashes),
            "answer_hashes": list(self.answer_hashes)
        }
        with open(filepath, 'w') as f:
            json.dump(state_dict, f)
    
    @classmethod
    def load_checkpoint(cls, filepath: str) -> 'GenerationState':
        """Load state from checkpoint file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
            state = cls(
                total_generated=state_dict["total_generated"],
                total_verified=state_dict["total_verified"],
                total_rejected=state_dict["total_rejected"],
                current_topic_idx=state_dict["current_topic_idx"],
                current_subtopic_idx=state_dict["current_subtopic_idx"],
                question_hashes=set(state_dict["question_hashes"]),
                answer_hashes=set(state_dict["answer_hashes"])
            )
            return state
        return cls()


# ============================================================================
# SECTION 5: MODEL LOADING
# ============================================================================

class ModelManager:
    """Manages the Hugging Face model with quantization."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the model with 4-bit quantization."""
        logger.info(f"🔄 Loading model: {self.config.model_name}")
        logger.info(f"🖥️ Device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Configure 4-bit quantization
        if self.config.use_quantization and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            logger.info("✅ Using 4-bit quantization (NF4)")
        else:
            quantization_config = None
            logger.info("ℹ️ Running without quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set padding side for generation
        self.tokenizer.padding_side = "left"
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Set model to eval mode
        self.model.eval()
        
        logger.info("✅ Model loaded successfully!")
        
        if self.device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1e9
            logger.info(f"📊 GPU Memory Used: {memory_used:.2f} GB")
    
    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        """Generate text from prompt using direct model.generate() - more efficient than pipeline."""
        try:
            # Format prompt for Mistral
            formatted_prompt = f"[INST] {prompt} [/INST]"
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode only the new tokens (skip the input)
            input_length = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:],
                skip_special_tokens=True
            ).strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""
    
    @torch.inference_mode()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate text for multiple prompts in a batch - more efficient for GPU."""
        try:
            # Format prompts for Mistral
            formatted_prompts = [f"[INST] {p} [/INST]" for p in prompts]
            
            # Tokenize with padding
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode each output
            results = []
            input_length = inputs["input_ids"].shape[1]
            for output in outputs:
                text = self.tokenizer.decode(
                    output[input_length:],
                    skip_special_tokens=True
                ).strip()
                results.append(text)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            return [""] * len(prompts)
    
    def clear_cache(self):
        """Clear GPU cache to prevent OOM."""
        if self.device == "cuda":
            torch.cuda.empty_cache()


# ============================================================================
# SECTION 6: AGENT IMPLEMENTATIONS
# ============================================================================

class BaseAgent:
    """Base class for all agents."""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        self.model = model_manager
        self.config = config
    
    def process(self, *args, **kwargs):
        raise NotImplementedError


class TopicPlannerAgent(BaseAgent):
    """Generates curriculum structure (uses predefined structure for consistency)."""
    
    def process(self) -> Dict[str, Any]:
        """Return the financial curriculum."""
        logger.info("📚 TopicPlanner: Using predefined financial curriculum")
        return FINANCIAL_CURRICULUM


class QuestionWriterAgent(BaseAgent):
    """Generates diverse educational questions for each topic."""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        super().__init__(model_manager, config)
        self.question_templates = {
            "definition": [
                "What is {concept}?",
                "Define {concept} in simple terms.",
                "What does {concept} mean in finance?"
            ],
            "conceptual": [
                "How does {concept} work?",
                "Explain the mechanism behind {concept}.",
                "What is the principle behind {concept}?"
            ],
            "comparison": [
                "What is the difference between {concept} and {related_concept}?",
                "How does {concept} compare to {related_concept}?",
                "When should you choose {concept} over {related_concept}?"
            ],
            "example": [
                "Can you give a real-world example of {concept}?",
                "How would {concept} apply in a practical scenario?",
                "What is a good illustration of {concept}?"
            ],
            "application": [
                "How can I apply {concept} to my personal finances?",
                "What are practical ways to use {concept}?",
                "How do investors use {concept}?"
            ],
            "misconception": [
                "What are common misconceptions about {concept}?",
                "What do most people get wrong about {concept}?",
                "What myths exist around {concept}?"
            ],
            "importance": [
                "Why is {concept} important for investors?",
                "What is the significance of {concept} in finance?",
                "Why should someone learn about {concept}?"
            ],
            "calculation": [
                "How do you calculate {concept}?",
                "What is the formula for {concept}?",
                "Walk me through calculating {concept}."
            ],
            "strategy": [
                "What strategies involve {concept}?",
                "How can {concept} be used strategically?",
                "What approaches work best with {concept}?"
            ],
            "risk": [
                "What are the risks associated with {concept}?",
                "What should I be careful about with {concept}?",
                "What are potential downsides of {concept}?"
            ]
        }
    
    def generate_questions(self, topic: str, subtopic: str, count: int = 5) -> List[Dict]:
        """Generate questions for a subtopic using LLM."""
        
        prompt = f"""You are a financial education expert. Generate {count} diverse educational questions about "{subtopic}" (under the topic "{topic}").

Requirements:
1. Questions should be clear and educational
2. Include different types: definition, conceptual, comparison, example, application
3. Range from beginner to advanced difficulty
4. Focus on teaching financial concepts

Output ONLY a numbered list of questions, nothing else.

Example format:
1. What is compound interest?
2. How does compound interest differ from simple interest?
3. Can you explain how compound interest affects long-term savings?

Generate {count} questions about {subtopic}:"""
        
        response = self.model.generate(prompt)
        questions = self._parse_questions(response, topic, subtopic)
        
        return questions
    
    def _parse_questions(self, response: str, topic: str, subtopic: str) -> List[Dict]:
        """Parse LLM response into structured questions."""
        questions = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Remove numbering
            if line and line[0].isdigit():
                # Remove "1.", "1)", "1:" etc.
                for sep in ['.', ')', ':']:
                    if sep in line[:3]:
                        line = line.split(sep, 1)[-1].strip()
                        break
            
            if line and '?' in line and len(line) > 15:
                question_type = self._classify_question_type(line)
                questions.append({
                    "topic": topic,
                    "subtopic": subtopic,
                    "question": line,
                    "question_type": question_type
                })
        
        return questions
    
    def _classify_question_type(self, question: str) -> str:
        """Classify the type of question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["what is", "define", "what does", "what are"]):
            return "definition"
        elif any(word in question_lower for word in ["how does", "how do", "explain", "mechanism"]):
            return "conceptual"
        elif any(word in question_lower for word in ["difference", "compare", "versus", "vs"]):
            return "comparison"
        elif any(word in question_lower for word in ["example", "scenario", "illustrat"]):
            return "example"
        elif any(word in question_lower for word in ["apply", "practical", "use"]):
            return "application"
        elif any(word in question_lower for word in ["misconception", "myth", "wrong"]):
            return "misconception"
        elif any(word in question_lower for word in ["why", "important", "significance"]):
            return "importance"
        elif any(word in question_lower for word in ["calculate", "formula", "compute"]):
            return "calculation"
        elif any(word in question_lower for word in ["strategy", "approach", "tactic"]):
            return "strategy"
        elif any(word in question_lower for word in ["risk", "danger", "careful", "downside"]):
            return "risk"
        else:
            return random.choice(QUESTION_TYPES)


class AnswerWriterAgent(BaseAgent):
    """Generates structured, educational answers."""
    
    def generate_answer(self, question_data: Dict) -> str:
        """Generate an educational answer for a question."""
        
        topic = question_data["topic"]
        subtopic = question_data["subtopic"]
        question = question_data["question"]
        
        prompt = f"""You are an expert financial educator. Answer the following question about {subtopic} ({topic}) in a clear, educational way.

Question: {question}

Requirements:
1. Provide a clear, accurate explanation
2. Use simple language suitable for learners
3. Include a practical example if applicable
4. Keep the answer between 150-400 words
5. Do NOT give personal financial advice
6. Focus on education, not recommendations

Answer:"""
        
        answer = self.model.generate(prompt)
        return answer.strip()


class ReviewerAgent(BaseAgent):
    """Reviews answers for quality and accuracy."""
    
    def review(self, qa_pair: QAPair) -> Tuple[str, str]:
        """Review a Q&A pair and return (status, feedback)."""
        
        # Quick quality checks
        if len(qa_pair.answer) < self.config.min_answer_length:
            return "rejected", "Answer too short"
        
        if qa_pair.question.lower() in qa_pair.answer.lower()[:100]:
            # Check if answer just repeats the question
            pass
        
        # Check for problematic content
        red_flags = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            "as an ai",
            "i apologize",
            "unfortunately"
        ]
        
        answer_lower = qa_pair.answer.lower()
        for flag in red_flags:
            if flag in answer_lower:
                return "needs_review", f"Contains uncertain language: {flag}"
        
        # Check for financial advice warnings
        advice_phrases = [
            "you should invest",
            "i recommend buying",
            "guaranteed returns",
            "risk-free",
            "get rich quick"
        ]
        
        for phrase in advice_phrases:
            if phrase in answer_lower:
                return "rejected", f"Contains potential financial advice: {phrase}"
        
        return "verified", "Passed quality checks"


class AntiHallucinationAgent(BaseAgent):
    """Flags potentially incorrect or unsupported claims."""
    
    def check(self, qa_pair: QAPair) -> Tuple[bool, str]:
        """Check for potential hallucinations."""
        
        answer = qa_pair.answer.lower()
        
        # Check for specific numerical claims that might be hallucinated
        suspicious_patterns = [
            "exactly ",
            "precisely ",
            "always returns",
            "never fails",
            "100%",
            "guaranteed"
        ]
        
        for pattern in suspicious_patterns:
            if pattern in answer:
                return False, f"Suspicious absolute claim: {pattern}"
        
        # Check for made-up statistics
        import re
        percentages = re.findall(r'\d+(?:\.\d+)?%', answer)
        if len(percentages) > 3:
            return False, "Too many specific percentages (potential hallucination)"
        
        return True, "No hallucination flags detected"


class DifficultyClassifierAgent(BaseAgent):
    """Classifies difficulty of Q&A pairs."""
    
    def classify(self, qa_pair: QAPair, topic_info: Dict) -> str:
        """Classify the difficulty of a Q&A pair."""
        
        # Get allowed difficulty range for the topic
        allowed_difficulties = topic_info.get("difficulty_range", ["beginner", "intermediate", "advanced"])
        
        question_lower = qa_pair.question.lower()
        answer_lower = qa_pair.answer.lower()
        
        # Advanced indicators
        advanced_terms = [
            "derivative", "hedge", "arbitrage", "volatility", "correlation",
            "beta", "alpha", "sharpe ratio", "monte carlo", "black-scholes",
            "stochastic", "discounted cash flow", "wacc", "capm"
        ]
        
        # Intermediate indicators
        intermediate_terms = [
            "portfolio", "diversification", "compound", "ratio", "margin",
            "leverage", "equity", "bond yield", "pe ratio", "market cap",
            "dividend yield", "asset allocation"
        ]
        
        # Count indicators
        advanced_count = sum(1 for term in advanced_terms if term in answer_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in answer_lower)
        
        if advanced_count >= 2 and "advanced" in allowed_difficulties:
            return "advanced"
        elif intermediate_count >= 2 or advanced_count >= 1:
            if "intermediate" in allowed_difficulties:
                return "intermediate"
        
        return "beginner" if "beginner" in allowed_difficulties else allowed_difficulties[0]


class DeduplicatorAgent(BaseAgent):
    """Removes duplicate or highly similar Q&A pairs."""
    
    def __init__(self, model_manager: ModelManager, config: Config):
        super().__init__(model_manager, config)
        self.embedding_model = None
        
        if EMBEDDINGS_AVAILABLE:
            try:
                logger.info("🔄 Loading embedding model for deduplication...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedding model loaded")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
    
    def compute_hash(self, text: str) -> str:
        """Compute hash of text for quick duplicate detection."""
        # Normalize text
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def is_duplicate(self, new_qa: QAPair, existing_hashes: set) -> bool:
        """Check if Q&A pair is a duplicate."""
        q_hash = self.compute_hash(new_qa.question)
        
        if q_hash in existing_hashes:
            return True
        
        return False
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self.embedding_model is None:
            # Fallback to simple word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0
        
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return similarity


# ============================================================================
# SECTION 7: MAIN GENERATOR CLASS
# ============================================================================

class FinancialDatasetGenerator:
    """Main orchestrator for dataset generation."""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = GenerationState.load_checkpoint(config.checkpoint_file)
        self.model_manager = None
        
        # Initialize agents (will be done after model loading)
        self.topic_planner = None
        self.question_writer = None
        self.answer_writer = None
        self.reviewer = None
        self.anti_hallucination = None
        self.difficulty_classifier = None
        self.deduplicator = None
        
        # Statistics
        self.stats = {
            "generated": 0,
            "verified": 0,
            "rejected": 0,
            "duplicates": 0,
            "hallucinations": 0
        }
    
    def initialize(self):
        """Initialize model and agents."""
        logger.info("=" * 60)
        logger.info("🚀 FINANCIAL EDUCATION DATASET GENERATOR")
        logger.info("=" * 60)
        
        # Load model
        self.model_manager = ModelManager(self.config)
        self.model_manager.load_model()
        
        # Initialize agents
        self.topic_planner = TopicPlannerAgent(self.model_manager, self.config)
        self.question_writer = QuestionWriterAgent(self.model_manager, self.config)
        self.answer_writer = AnswerWriterAgent(self.model_manager, self.config)
        self.reviewer = ReviewerAgent(self.model_manager, self.config)
        self.anti_hallucination = AntiHallucinationAgent(self.model_manager, self.config)
        self.difficulty_classifier = DifficultyClassifierAgent(self.model_manager, self.config)
        self.deduplicator = DeduplicatorAgent(self.model_manager, self.config)
        
        logger.info("✅ All agents initialized")
        
        # Resume from checkpoint if exists
        if self.state.total_generated > 0:
            logger.info(f"📂 Resuming from checkpoint: {self.state.total_generated} pairs already generated")
    
    def generate_id(self, topic: str, subtopic: str, idx: int) -> str:
        """Generate unique ID for Q&A pair."""
        return f"fin_edu_{topic[:3]}_{subtopic[:3]}_{idx}_{int(time.time())}"
    
    def process_single_qa(self, question_data: Dict, topic_info: Dict) -> Optional[QAPair]:
        """Process a single Q&A through the pipeline."""
        
        try:
            # Step 1: Generate answer
            answer = self.answer_writer.generate_answer(question_data)
            
            if not answer or len(answer) < self.config.min_answer_length:
                self.stats["rejected"] += 1
                return None
            
            # Step 2: Create Q&A pair
            qa_id = self.generate_id(
                question_data["topic"],
                question_data["subtopic"],
                self.state.total_generated
            )
            
            qa_pair = QAPair(
                id=qa_id,
                topic=question_data["topic"],
                subtopic=question_data["subtopic"],
                question=question_data["question"],
                answer=answer,
                difficulty="pending",
                question_type=question_data["question_type"]
            )
            
            # Step 3: Check for duplicates
            if self.deduplicator.is_duplicate(qa_pair, self.state.question_hashes):
                self.stats["duplicates"] += 1
                return None
            
            # Step 4: Anti-hallucination check
            is_valid, reason = self.anti_hallucination.check(qa_pair)
            if not is_valid:
                self.stats["hallucinations"] += 1
                return None
            
            # Step 5: Review
            review_status, feedback = self.reviewer.review(qa_pair)
            qa_pair.review_status = review_status
            
            if review_status == "rejected":
                self.stats["rejected"] += 1
                return None
            
            # Step 6: Classify difficulty
            qa_pair.difficulty = self.difficulty_classifier.classify(qa_pair, topic_info)
            
            # Update state
            self.state.question_hashes.add(self.deduplicator.compute_hash(qa_pair.question))
            self.state.answer_hashes.add(self.deduplicator.compute_hash(qa_pair.answer))
            
            self.stats["generated"] += 1
            if review_status == "verified":
                self.stats["verified"] += 1
            
            return qa_pair
            
        except Exception as e:
            logger.error(f"Error processing Q&A: {e}")
            return None
    
    def save_batch(self, qa_pairs: List[QAPair]):
        """Save a batch of Q&A pairs to file."""
        mode = 'a' if os.path.exists(self.config.output_file) else 'w'
        
        with open(self.config.output_file, mode, encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(qa.to_jsonl() + '\n')
        
        # Save checkpoint
        self.state.total_generated = self.stats["generated"]
        self.state.total_verified = self.stats["verified"]
        self.state.total_rejected = self.stats["rejected"]
        self.state.save_checkpoint(self.config.checkpoint_file)
    
    def print_progress(self):
        """Print current progress."""
        total = self.stats["generated"]
        target = self.config.target_dataset_size
        percentage = (total / target) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 PROGRESS: {total:,} / {target:,} ({percentage:.1f}%)")
        print(f"   ✅ Verified: {self.stats['verified']:,}")
        print(f"   ❌ Rejected: {self.stats['rejected']:,}")
        print(f"   🔄 Duplicates: {self.stats['duplicates']:,}")
        print(f"   ⚠️ Hallucinations: {self.stats['hallucinations']:,}")
        print(f"{'='*60}\n")
    
    def generate_dataset(self):
        """Main generation loop."""
        
        # Get curriculum
        curriculum = self.topic_planner.process()
        topics = list(curriculum.keys())
        
        # Calculate questions per subtopic to reach target
        total_subtopics = sum(len(info["subtopics"]) for info in curriculum.values())
        questions_per_subtopic = max(10, self.config.target_dataset_size // total_subtopics)
        
        logger.info(f"📖 Curriculum: {len(topics)} topics, {total_subtopics} subtopics")
        logger.info(f"🎯 Target: ~{questions_per_subtopic} Q&A per subtopic")
        
        batch = []
        generation_count = 0
        
        # Main generation loop
        try:
            for topic_idx, topic in enumerate(topics):
                topic_info = curriculum[topic]
                subtopics = topic_info["subtopics"]
                
                for subtopic_idx, subtopic in enumerate(subtopics):
                    logger.info(f"\n📝 Processing: {topic} > {subtopic}")
                    
                    # Generate questions for this subtopic
                    for q_batch in range(0, questions_per_subtopic, self.config.batch_size):
                        # Generate a batch of questions
                        questions = self.question_writer.generate_questions(
                            topic, subtopic, 
                            count=min(self.config.batch_size, questions_per_subtopic - q_batch)
                        )
                        
                        # Process each question
                        for question_data in tqdm(questions, desc=f"{subtopic[:20]}...", leave=False):
                            qa_pair = self.process_single_qa(question_data, topic_info)
                            
                            if qa_pair:
                                batch.append(qa_pair)
                                generation_count += 1
                            
                            # Save periodically
                            if len(batch) >= self.config.save_interval:
                                self.save_batch(batch)
                                batch = []
                                self.print_progress()
                            
                            # Clear GPU cache periodically
                            if generation_count % self.config.clear_cache_interval == 0:
                                self.model_manager.clear_cache()
                            
                            # Check if target reached
                            if self.stats["generated"] >= self.config.target_dataset_size:
                                logger.info("🎉 Target dataset size reached!")
                                break
                        
                        if self.stats["generated"] >= self.config.target_dataset_size:
                            break
                    
                    if self.stats["generated"] >= self.config.target_dataset_size:
                        break
                
                if self.stats["generated"] >= self.config.target_dataset_size:
                    break
        
        except KeyboardInterrupt:
            logger.info("\n⚠️ Generation interrupted by user")
        
        finally:
            # Save remaining batch
            if batch:
                self.save_batch(batch)
            
            self.print_final_stats()
    
    def print_final_stats(self):
        """Print final statistics."""
        print("\n" + "=" * 60)
        print("🏁 GENERATION COMPLETE")
        print("=" * 60)
        print(f"📊 Total Q&A pairs generated: {self.stats['generated']:,}")
        print(f"✅ Verified: {self.stats['verified']:,}")
        print(f"❌ Rejected: {self.stats['rejected']:,}")
        print(f"🔄 Duplicates removed: {self.stats['duplicates']:,}")
        print(f"⚠️ Hallucinations caught: {self.stats['hallucinations']:,}")
        print(f"\n📁 Output file: {self.config.output_file}")
        
        if os.path.exists(self.config.output_file):
            file_size = os.path.getsize(self.config.output_file) / (1024 * 1024)
            print(f"📦 File size: {file_size:.2f} MB")
        
        print("=" * 60)


# ============================================================================
# SECTION 8: UTILITY FUNCTIONS
# ============================================================================

def validate_dataset(filepath: str) -> Dict[str, Any]:
    """Validate the generated dataset."""
    
    print(f"\n🔍 Validating dataset: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return {}
    
    stats = {
        "total_pairs": 0,
        "by_topic": defaultdict(int),
        "by_difficulty": defaultdict(int),
        "by_status": defaultdict(int),
        "avg_question_length": 0,
        "avg_answer_length": 0
    }
    
    q_lengths = []
    a_lengths = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                qa = json.loads(line.strip())
                stats["total_pairs"] += 1
                stats["by_topic"][qa.get("topic", "unknown")] += 1
                stats["by_difficulty"][qa.get("difficulty", "unknown")] += 1
                stats["by_status"][qa.get("review_status", "unknown")] += 1
                q_lengths.append(len(qa.get("question", "")))
                a_lengths.append(len(qa.get("answer", "")))
            except json.JSONDecodeError:
                continue
    
    if q_lengths:
        stats["avg_question_length"] = sum(q_lengths) / len(q_lengths)
    if a_lengths:
        stats["avg_answer_length"] = sum(a_lengths) / len(a_lengths)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total pairs: {stats['total_pairs']:,}")
    print(f"   Avg question length: {stats['avg_question_length']:.0f} chars")
    print(f"   Avg answer length: {stats['avg_answer_length']:.0f} chars")
    print(f"\n📚 By Topic:")
    for topic, count in sorted(stats["by_topic"].items(), key=lambda x: -x[1]):
        print(f"   {topic}: {count:,}")
    print(f"\n📈 By Difficulty:")
    for diff, count in stats["by_difficulty"].items():
        print(f"   {diff}: {count:,}")
    
    return stats


def sample_dataset(filepath: str, n: int = 5):
    """Print sample entries from the dataset."""
    
    print(f"\n📋 Sample entries from {filepath}:")
    print("=" * 60)
    
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line.strip()))
    
    for i, qa in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Topic: {qa.get('topic')} > {qa.get('subtopic')}")
        print(f"Difficulty: {qa.get('difficulty')}")
        print(f"Question: {qa.get('question')}")
        answer = qa.get('answer', '')
        print(f"Answer: {answer[:300]}{'...' if len(answer) > 300 else ''}")
        print()


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def download_dataset(filepath: str):
    """Download the dataset file in Google Colab."""
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    if is_colab():
        try:
            from google.colab import files
            print(f"\n📥 Initiating download for: {filepath}")
            files.download(filepath)
            print("✅ Download started! Check your browser downloads.")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("💡 You can manually download from the Files panel on the left.")
            return False
    else:
        print(f"\n📁 Not running in Colab. File saved at: {os.path.abspath(filepath)}")
        return True


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║     FINANCIAL EDUCATION DATASET GENERATOR                     ║
    ║     Multi-Agent System for Q&A Generation                     ║
    ║     Optimized for Google Colab                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("⚠️ No GPU detected. Running on CPU (will be slower)")
    
    # Initialize configuration
    config = Config()
    
    # Allow user to modify target size
    print(f"\n📊 Current settings:")
    print(f"   Target dataset size: {config.target_dataset_size:,}")
    print(f"   Model: {config.model_name}")
    print(f"   Quantization: {'4-bit (NF4)' if config.use_quantization else 'None'}")
    print(f"   Output file: {config.output_file}")
    
    # Create generator and run
    generator = FinancialDatasetGenerator(config)
    generator.initialize()
    generator.generate_dataset()
    
    # Validate output
    if os.path.exists(config.output_file):
        validate_dataset(config.output_file)
        sample_dataset(config.output_file, n=3)
        
        print("\n✅ Generation complete! Your dataset is ready.")
        print(f"📁 File location: {config.output_file}")
        
        # Auto-download in Google Colab
        if is_colab():
            print("\n" + "="*60)
            print("📥 DOWNLOADING DATASET...")
            print("="*60)
            download_dataset(config.output_file)
            
            # Also offer to download checkpoint if exists
            if os.path.exists(config.checkpoint_file):
                print("\n💾 Checkpoint file also available for download.")
        else:
            print(f"\n💡 To download in Colab, use: files.download('{config.output_file}')")
    else:
        print("\n❌ No output file generated. Check logs for errors.")


if __name__ == "__main__":
    main()
