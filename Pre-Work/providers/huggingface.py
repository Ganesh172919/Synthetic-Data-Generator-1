"""
HuggingFace Provider
====================
Runs local models via the Transformers library.
GPU recommended for reasonable performance.
"""

import os
import time
from typing import List

from .base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
    ProviderError,
)
from .factory import register_provider


@register_provider('huggingface')
class HuggingFaceProvider(BaseProvider):

    DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._loaded_model_name = None

    @property
    def name(self) -> str:
        return 'huggingface'

    @property
    def requires_api_key(self) -> bool:
        return False

    def _load_model(self, model_name: str):
        if self._loaded_model_name == model_name and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            raise ProviderError(
                "transformers/torch not installed. Run: pip install transformers torch accelerate bitsandbytes",
                provider='huggingface',
            )

        try:
            quantization_config = None
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            model_kwargs = {"trust_remote_code": True, "device_map": "auto"}
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

            self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            self._loaded_model_name = model_name
        except Exception as exc:
            raise ProviderError(
                f"Failed to load model {model_name}: {exc}",
                provider='huggingface',
            ) from exc

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            import torch
        except ImportError:
            raise ProviderError("torch not installed", provider='huggingface')

        model_name = request.model or self.DEFAULT_MODEL
        self._load_model(model_name)

        try:
            messages = []
            if request.system_prompt:
                messages.append({"role": "user", "content": f"{request.system_prompt}\n\n{request.prompt}"})
            else:
                messages.append({"role": "user", "content": request.prompt})

            input_ids = self._tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids,
                    max_new_tokens=request.max_new_tokens,
                    temperature=max(request.temperature, 0.01),
                    do_sample=request.temperature > 0,
                    top_p=0.95,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            new_tokens = outputs[0][input_ids.shape[-1]:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            return GenerationResponse(
                text=text,
                tokens_used=len(new_tokens),
                model=model_name,
                finish_reason='stop',
            )
        except Exception as exc:
            raise ProviderError(str(exc), provider='huggingface') from exc

    def health_check(self) -> ProviderHealth:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device = torch.cuda.get_device_name(0) if cuda_available else "CPU"
            return ProviderHealth(
                status='healthy',
                message=f'HuggingFace provider ready ({device})',
                models_available=[self.DEFAULT_MODEL],
            )
        except ImportError:
            return ProviderHealth(
                status='unavailable',
                message='torch/transformers not installed',
            )

    def get_models(self) -> List[str]:
        return [self.DEFAULT_MODEL]
