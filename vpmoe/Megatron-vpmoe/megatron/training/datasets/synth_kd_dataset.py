# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Dict, Optional

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, _get_ltor_masks_and_position_ids
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.object_storage_utils import ObjectStorageConfig, is_object_storage_path
from megatron.core.datasets.utils import Split


TOKENS_SUFFIX = "_tokens"
LOSSMASK_SUFFIX = "_lossmask"
SPAN_SUFFIX = "_span"


class SynthKDDataset(GPTDataset):
    """GPTDataset variant that loads aligned KD loss/span masks."""

    def __init__(
        self,
        indexed_dataset: IndexedDataset,
        dataset_path: Optional[str],
        indexed_indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(
            indexed_dataset, dataset_path, indexed_indices, num_samples, index_split, config
        )
        if dataset_path is None:
            raise ValueError("SynthKDDataset requires a dataset path prefix.")

        self.lossmask_dataset = self._build_sidecar_dataset(dataset_path, LOSSMASK_SUFFIX)
        self.span_dataset = self._build_sidecar_dataset(dataset_path, SPAN_SUFFIX)
        self._validate_sidecars()

    def _build_sidecar_dataset(self, dataset_path: str, suffix: str) -> IndexedDataset:
        if not dataset_path.endswith(TOKENS_SUFFIX):
            raise ValueError(
                f"SynthKDDataset expects token prefixes to end with {TOKENS_SUFFIX}: {dataset_path}"
            )
        base_prefix = dataset_path[: -len(TOKENS_SUFFIX)]
        path_prefix = f"{base_prefix}{suffix}"

        if is_object_storage_path(path_prefix):
            assert self.config.object_storage_cache_path is not None
            return IndexedDataset(
                path_prefix,
                multimodal=False,
                mmap=self.config.mmap_bin_files,
                object_storage_config=ObjectStorageConfig(
                    path_to_idx_cache=self.config.object_storage_cache_path
                ),
            )

        return IndexedDataset(
            path_prefix,
            multimodal=False,
            mmap=self.config.mmap_bin_files,
            fast_cache_load=self.config.fast_cache_load,
        )

    def _validate_sidecars(self) -> None:
        if self.dataset.sequence_lengths.shape[0] != self.lossmask_dataset.sequence_lengths.shape[0]:
            raise RuntimeError("Lossmask dataset length does not match tokens dataset.")
        if self.dataset.sequence_lengths.shape[0] != self.span_dataset.sequence_lengths.shape[0]:
            raise RuntimeError("Span dataset length does not match tokens dataset.")
        if self.dataset.document_indices[-1] != self.lossmask_dataset.document_indices[-1]:
            raise RuntimeError("Lossmask document indices do not match tokens dataset.")
        if self.dataset.document_indices[-1] != self.span_dataset.document_indices[-1]:
            raise RuntimeError("Span document indices do not match tokens dataset.")

    def _ensure_indices_loaded(self) -> None:
        if self.shuffle_index is None:
            self.shuffle_index = np.load(
                self.path_to_shuffle_index, allow_pickle=True, mmap_mode="r"
            )
            self.sample_index = np.load(
                self.path_to_sample_index, allow_pickle=True, mmap_mode="r"
            )
            self.document_index = np.load(
                self.path_to_document_index, allow_pickle=True, mmap_mode="r"
            )

    def _query_sample(self, idx: int, dataset: IndexedDataset, pad_value: int) -> np.ndarray:
        self._ensure_indices_loaded()

        idx = self.shuffle_index[idx]
        doc_index_beg, doc_index_beg_offset = self.sample_index[idx]
        doc_index_end, doc_index_end_offset = self.sample_index[idx + 1]

        sample_parts = []
        if doc_index_beg == doc_index_end:
            sample_parts.append(
                dataset.get(
                    self.document_index[doc_index_beg],
                    offset=doc_index_beg_offset,
                    length=doc_index_end_offset
                    - doc_index_beg_offset
                    + self.config.add_extra_token_to_sequence,
                )
            )
        else:
            for i in range(doc_index_beg, doc_index_end + 1):
                offset = 0 if i > doc_index_beg else doc_index_beg_offset
                length = (
                    None
                    if i < doc_index_end
                    else doc_index_end_offset + self.config.add_extra_token_to_sequence
                )
                sample_parts.append(
                    dataset.get(self.document_index[i], offset=offset, length=length)
                )

        length = sum(map(len, sample_parts))
        target_len = self.config.sequence_length + self.config.add_extra_token_to_sequence
        if length < target_len:
            sample_parts.append([pad_value] * (target_len - length))

        return np.concatenate(sample_parts, dtype=np.int64)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        if idx is None:
            idx = 0
            zero_masks = True
        else:
            zero_masks = False

        text = self._query_sample(idx, self.dataset, self._pad_token_id)
        loss_mask = self._query_sample(idx, self.lossmask_dataset, 0)
        span_id = self._query_sample(idx, self.span_dataset, 0)

        text = torch.from_numpy(text).long()
        loss_mask = torch.from_numpy(loss_mask).long()
        span_id = torch.from_numpy(span_id).long()

        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
            loss_mask = loss_mask[:-1].contiguous()
            span_id = span_id[:-1].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id
            loss_mask = loss_mask.contiguous()
            span_id = span_id.contiguous()
            loss_mask[-1] = 0
            span_id[-1] = 0

        if (
            not self.masks_and_position_ids_are_cacheable
            or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, _, position_ids = _get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            position_ids = self.cached_position_ids

        loss_mask = loss_mask.float()
        loss_mask[labels == self._pad_token_id] = 0.0
        span_id[labels == self._pad_token_id] = 0

        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        if zero_masks:
            loss_mask = torch.zeros_like(loss_mask)
            span_id = torch.zeros_like(span_id)

        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "span_id": span_id,
            }
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "span_id": span_id,
        }
