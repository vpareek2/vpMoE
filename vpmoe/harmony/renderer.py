"""Harmony renderer backed by the official openai-harmony package."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    RenderConversationConfig,
    Role,
    load_harmony_encoding,
)


ALLOWED_ROLES = {"system", "developer", "user", "assistant"}
ASSISTANT_CHANNELS = {"analysis", "final"}


@dataclass(frozen=True)
class _HarmonyState:
    encoding: object
    return_token: int
    eod_token: int
    config: RenderConversationConfig


_STATE: _HarmonyState | None = None


def _resolve_tiktoken_base() -> Path:
    base_dir = os.getenv("TIKTOKEN_ENCODINGS_BASE")
    if not base_dir:
        vocab_path = os.getenv("O200K_HARMONY_VOCAB_PATH")
        if vocab_path:
            base_dir = str(Path(vocab_path).parent)
            os.environ["TIKTOKEN_ENCODINGS_BASE"] = base_dir
    if not base_dir:
        raise ValueError(
            "Set TIKTOKEN_ENCODINGS_BASE to the directory containing "
            "o200k_base.tiktoken (or set O200K_HARMONY_VOCAB_PATH)."
        )

    base_path = Path(base_dir)
    vocab_file = base_path / "o200k_base.tiktoken"
    if not vocab_file.exists():
        raise ValueError(
            f"Missing o200k_base.tiktoken in {base_path}. "
            "Provide a local tokenizer file and set TIKTOKEN_ENCODINGS_BASE."
        )
    return base_path


def _get_state() -> _HarmonyState:
    global _STATE
    if _STATE is None:
        _resolve_tiktoken_base()
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        return_token = encoding.encode("<|return|>", allowed_special={"<|return|>"})[0]
        eod_token = encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        config = RenderConversationConfig(auto_drop_analysis=False)
        _STATE = _HarmonyState(
            encoding=encoding,
            return_token=return_token,
            eod_token=eod_token,
            config=config,
        )
    return _STATE


def render_messages(messages: Iterable[dict]) -> Tuple[List[int], List[int], List[int]]:
    """Render Harmony messages into tokens, loss masks, and span ids."""
    filtered: List[dict] = []
    has_assistant = False
    for msg in messages:
        role = msg.get("role")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Unsupported role: {role}")
        if role == "assistant":
            has_assistant = True
        filtered.append(msg)

    if not filtered:
        raise ValueError("No messages to render.")
    if not has_assistant:
        raise ValueError("No assistant messages found in conversation.")

    state = _get_state()
    oa_messages = [Message.from_dict(msg) for msg in filtered]
    convo = Conversation.from_messages(oa_messages)

    rendered_tokens = state.encoding.render_conversation_for_training(
        convo, config=state.config
    )

    tokens: List[int] = []
    loss_mask: List[int] = []
    span_id: List[int] = []

    last_index = len(oa_messages) - 1
    for idx, msg in enumerate(oa_messages):
        msg_tokens = state.encoding.render(msg)
        role = msg.author.role

        if role == Role.ASSISTANT:
            if msg.channel not in ASSISTANT_CHANNELS:
                raise ValueError(f"Assistant message missing/invalid channel: {msg.channel!r}")
            if idx == last_index and msg.channel == "final":
                msg_tokens = msg_tokens[:-1] + [state.return_token]
            mask_value = 1
            span_value = 1 if msg.channel == "analysis" else 2
        elif role in (Role.USER, Role.SYSTEM, Role.DEVELOPER):
            mask_value = 0
            span_value = 0
        else:
            raise ValueError(f"Unsupported role after filtering: {role}")

        tokens.extend(msg_tokens)
        loss_mask.extend([mask_value] * len(msg_tokens))
        span_id.extend([span_value] * len(msg_tokens))

    if tokens != rendered_tokens:
        raise ValueError("Message render does not match conversation render.")

    tokens.append(state.eod_token)
    loss_mask.append(0)
    span_id.append(0)

    return tokens, loss_mask, span_id
