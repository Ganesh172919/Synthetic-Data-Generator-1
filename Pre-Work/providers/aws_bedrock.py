"""
AWS Bedrock Provider
====================
Uses AWS Bedrock for multi-model access (Claude, Llama, Mistral, etc.).
Requires AWS credentials configured via environment or ~/.aws/credentials.
"""

import os
import time
import json
from typing import List

from .base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderError,
)
from .factory import register_provider


@register_provider('aws_bedrock')
class AWSBedrockProvider(BaseProvider):

    AVAILABLE_MODELS = [
        'anthropic.claude-3-5-sonnet-20241022-v2:0',
        'anthropic.claude-3-5-haiku-20241022-v1:0',
        'meta.llama3-1-70b-instruct-v1:0',
        'meta.llama3-1-8b-instruct-v1:0',
        'mistral.mistral-large-2402-v1:0',
    ]

    def __init__(self):
        self._region = os.environ.get('AWS_REGION', 'us-east-1')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client('bedrock-runtime', region_name=self._region)
            except ImportError:
                raise ProviderError(
                    "boto3 not installed. Run: pip install boto3",
                    provider='aws_bedrock',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'aws_bedrock'

    @property
    def requires_api_key(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        model_id = request.model or 'anthropic.claude-3-5-sonnet-20241022-v2:0'

        try:
            # Build request body based on model family
            if 'anthropic' in model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                    "messages": [{"role": "user", "content": request.prompt}],
                }
                if request.system_prompt:
                    body["system"] = request.system_prompt
            elif 'meta.llama' in model_id:
                prompt_text = request.prompt
                if request.system_prompt:
                    prompt_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{request.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{request.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                body = {
                    "prompt": prompt_text,
                    "max_gen_len": request.max_new_tokens,
                    "temperature": request.temperature,
                }
            elif 'mistral' in model_id:
                body = {
                    "prompt": request.prompt,
                    "max_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                }
            else:
                body = {
                    "prompt": request.prompt,
                    "max_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                }

            response = client.invoke_model(
                modelId=model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(body),
            )

            result = json.loads(response['body'].read())

            # Extract text based on model family
            if 'anthropic' in model_id:
                text = ""
                for block in result.get('content', []):
                    if block.get('type') == 'text':
                        text += block.get('text', '')
                tokens = result.get('usage', {}).get('input_tokens', 0) + result.get('usage', {}).get('output_tokens', 0)
            elif 'meta.llama' in model_id:
                text = result.get('generation', '')
                tokens = result.get('prompt_token_count', 0) + result.get('generation_token_count', 0)
            elif 'mistral' in model_id:
                outputs = result.get('outputs', [])
                text = outputs[0].get('text', '') if outputs else ''
                tokens = 0
            else:
                text = str(result)
                tokens = 0

            return GenerationResponse(
                text=text,
                tokens_used=tokens,
                model=model_id,
                finish_reason='stop',
                raw=result,
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if 'throttl' in error_msg or 'rate' in error_msg or '429' in error_msg:
                raise ProviderRateLimitError(str(exc), provider='aws_bedrock') from exc
            if 'access' in error_msg or 'auth' in error_msg or '401' in error_msg or '403' in error_msg:
                raise ProviderAuthError(str(exc), provider='aws_bedrock') from exc
            raise ProviderError(str(exc), provider='aws_bedrock') from exc

    def health_check(self) -> ProviderHealth:
        try:
            import boto3
            sts = boto3.client('sts', region_name=self._region)
            identity = sts.get_caller_identity()
            return ProviderHealth(
                status='healthy',
                message=f'AWS authenticated as {identity.get("Arn", "unknown")} in {self._region}',
                models_available=self.AVAILABLE_MODELS,
            )
        except ImportError:
            return ProviderHealth(
                status='unavailable',
                message='boto3 not installed',
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=f'AWS auth failed: {exc}',
            )

    def get_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
