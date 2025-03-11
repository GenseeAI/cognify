from litellm import Usage
from openai.types.completion_usage import (
    CompletionUsage,
    PromptTokensDetails,
    CompletionTokensDetails,
)
from dataclasses import dataclass
from typing import List, Optional, Dict


@dataclass
class ResponseMetadata:
    model: str
    cost: float
    usage: Usage


def aggregate_usages(usages: List[Usage]) -> CompletionUsage:
    aggregated_usage = CompletionUsage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(
            audio_tokens=0, reasoning_tokens=0
        ),
    )
    for usage in usages:
        aggregated_usage.prompt_tokens += usage.get("prompt_tokens", 0)
        aggregated_usage.completion_tokens += usage.get("completion_tokens", 0)
        aggregated_usage.total_tokens += usage.get("total_tokens", 0)

        # token details
        prompt_token_details = usage.get("prompt_tokens_details", None)
        if prompt_token_details:
            aggregated_usage.prompt_tokens_details += prompt_token_details.get(
                "audio_tokens", 0
            )
            aggregated_usage.prompt_tokens_details += prompt_token_details.get(
                "cached_tokens", 0
            )

        completion_token_details = usage.get("completion_tokens_details", None)
        if completion_token_details:
            aggregated_usage.completion_tokens_details += completion_token_details.get(
                "audio_tokens", 0
            )
            aggregated_usage.completion_tokens_details += completion_token_details.get(
                "reasoning_tokens", 0
            )
    return aggregated_usage


@dataclass
class StepInfo:
    filled_inputs_dict: Dict[str, str]  # input name -> input value
    output: str
    rationale: Optional[str] = None
