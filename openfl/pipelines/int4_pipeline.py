# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Int4Pipeline module."""

import gzip as gz
from typing import Optional

import numpy as np

from openfl.pipelines.pipeline import TransformationPipeline, Transformer


class Int4Transformer(Transformer):
    """Transformer for block-wise INT4 quantization with sub-byte packing.

    Quantizes floating-point arrays into 4-bit signed integers in [-8, 7] using
    per-block dynamic scaling. Packs two 4-bit values per byte (nibble packing)
    to achieve up to 8x communication payload reduction for edge model exchange.

    Attributes:
        block_size (Optional[int]): Number of elements per quantization block.
            If None, scales the entire tensor with a single dynamic scale factor.
        stochastic (bool): Whether to use stochastic rounding for unbiased estimation.
        use_gzip (bool): Whether to apply GZIP compression over the packed bytes.
        lossy (bool): Indicates if the transformer is lossy. Always True.
    """

    def __init__(
        self,
        block_size: Optional[int] = 64,
        stochastic: bool = False,
        use_gzip: bool = False,
    ):
        """Initialize Int4Transformer.

        Args:
            block_size (Optional[int], optional): Number of elements per block.
                Defaults to 64.
            stochastic (bool, optional): Whether to use stochastic rounding.
                Defaults to False.
            use_gzip (bool, optional): Whether to apply GZIP on packed bytes.
                Defaults to False.

        Raises:
            ValueError: If block_size is invalid (not a positive integer or None).
        """
        if block_size is not None:
            if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
                raise ValueError(f"block_size must be a positive integer or None, got {block_size}")
        if not isinstance(stochastic, bool):
            raise ValueError(f"stochastic must be a boolean, got {type(stochastic)}")
        if not isinstance(use_gzip, bool):
            raise ValueError(f"use_gzip must be a boolean, got {type(use_gzip)}")

        self.block_size = block_size
        self.stochastic = stochastic
        self.use_gzip = use_gzip
        self.lossy = True

    def forward(self, data: np.ndarray, **kwargs):
        """Quantize input array to packed INT4 bytes and metadata.

        Args:
            data (np.ndarray): The floating-point numpy array to transform.
            **kwargs: Additional parameters.

        Returns:
            Tuple[bytes, dict]: Packed bytes and metadata dictionary.

        Raises:
            ValueError: If input contains non-finite values (NaN or Inf).
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data)}")

        if not np.all(np.isfinite(data)):
            raise ValueError("Input array contains non-finite values (NaN or Inf).")

        orig_shape = list(data.shape)
        if data.size == 0:
            metadata = {
                "int_to_float": {},
                "int_list": [len(orig_shape)] + orig_shape + [0, 0, 0],
                "bool_list": [True, self.stochastic, self.use_gzip],
            }
            return b"", metadata

        flat = data.astype(np.float32).flatten()
        n = flat.size

        if self.block_size is None or self.block_size >= n:
            block_size = n
            pad_len = 0
            padded = flat
            n_blocks = 1
        else:
            block_size = self.block_size
            pad_len = (block_size - (n % block_size)) % block_size
            padded = np.pad(flat, (0, pad_len), mode="constant") if pad_len > 0 else flat
            n_blocks = len(padded) // block_size

        blocks = padded.reshape(n_blocks, block_size)
        max_abs = np.max(np.abs(blocks), axis=1)
        zero_mask = max_abs == 0.0

        # Symmetrically map to [-8, 7]
        scales = max_abs / 7.0
        safe_scales = np.where(zero_mask, 1.0, scales)

        normalized = blocks / safe_scales[:, np.newaxis]
        if self.stochastic:
            normalized += np.random.uniform(-0.5, 0.5, size=normalized.shape)

        q = np.where(
            zero_mask[:, np.newaxis],
            0,
            np.clip(np.round(normalized), -8, 7),
        ).astype(np.int8)

        q_flat = q.flatten()
        sub_pad = len(q_flat) % 2
        if sub_pad > 0:
            q_flat = np.pad(q_flat, (0, sub_pad), mode="constant")

        # Offset to unsigned [0, 15] for nibble packing
        q_u = (q_flat + 8).astype(np.uint8)
        high = (q_u[0::2] & 0x0F) << 4
        low = q_u[1::2] & 0x0F
        packed = (high | low).astype(np.uint8)
        packed_bytes = packed.tobytes()

        if self.use_gzip:
            data_bytes = gz.compress(packed_bytes)
        else:
            data_bytes = packed_bytes

        int_to_float = {int(i): float(s) for i, s in enumerate(scales)}
        int_list = [
            len(orig_shape),
            *orig_shape,
            int(self.block_size or 0),
            int(pad_len),
            int(sub_pad),
        ]
        bool_list = [bool(self.lossy), bool(self.stochastic), bool(self.use_gzip)]

        metadata = {
            "int_to_float": int_to_float,
            "int_list": int_list,
            "bool_list": bool_list,
        }
        return data_bytes, metadata

    def backward(self, data: bytes, metadata: dict, **kwargs) -> np.ndarray:
        """Dequantize packed INT4 bytes back to float32 numpy array.

        Args:
            data (bytes): The packed bytes to reconstruct.
            metadata (dict): Metadata containing shape, block size, and scales.
            **kwargs: Additional parameters.

        Returns:
            np.ndarray: Reconstructed float32 numpy array.
        """
        int_list = metadata["int_list"]
        shape_len = int_list[0]
        orig_shape = tuple(int_list[1 : 1 + shape_len])

        total_elements = 1
        for dim in orig_shape:
            total_elements *= dim
        if total_elements == 0:
            return np.empty(orig_shape, dtype=np.float32)

        bs = int_list[1 + shape_len]
        pad_len = int_list[2 + shape_len]
        sub_pad = int_list[3 + shape_len]
        block_size = bs if bs > 0 else None

        bool_list = metadata.get("bool_list", [])
        use_gzip = bool_list[2] if len(bool_list) >= 3 else False

        raw_bytes = gz.decompress(data) if use_gzip else data
        packed = np.frombuffer(raw_bytes, dtype=np.uint8)

        high = (packed >> 4) & 0x0F
        low = packed & 0x0F
        unpacked = np.empty(len(packed) * 2, dtype=np.uint8)
        unpacked[0::2] = high
        unpacked[1::2] = low

        if sub_pad > 0:
            unpacked = unpacked[:-sub_pad]

        q = unpacked.astype(np.int8) - 8

        int_to_float = metadata.get("int_to_float", {})
        if block_size is None or block_size >= len(q):
            scale = int_to_float.get(0, 0.0)
            flat = q.astype(np.float32) * scale
        else:
            n_blocks = len(q) // block_size
            scales = np.array(
                [int_to_float.get(i, 0.0) for i in range(n_blocks)],
                dtype=np.float32,
            )
            blocks = q.reshape(n_blocks, block_size).astype(np.float32)
            flat = (blocks * scales[:, np.newaxis]).flatten()

        if pad_len > 0:
            flat = flat[:-pad_len]

        return flat.reshape(orig_shape).astype(np.float32)


class Int4Pipeline(TransformationPipeline):
    """Pipeline for block-wise INT4 compression with sub-byte packing.

    Attributes:
        block_size (Optional[int]): Block size for dynamic scaling.
        stochastic (bool): Whether stochastic rounding is enabled.
        use_gzip (bool): Whether GZIP is applied over packed bytes.
    """

    def __init__(
        self,
        block_size: Optional[int] = 64,
        stochastic: bool = False,
        use_gzip: bool = False,
        **kwargs,
    ):
        """Initialize Int4Pipeline.

        Args:
            block_size (Optional[int], optional): Block size for scaling.
                Defaults to 64.
            stochastic (bool, optional): Whether to use stochastic rounding.
                Defaults to False.
            use_gzip (bool, optional): Whether to apply GZIP compression.
                Defaults to False.
            **kwargs: Additional keyword arguments for TransformationPipeline.
        """
        self.block_size = block_size
        self.stochastic = stochastic
        self.use_gzip = use_gzip
        transformer = Int4Transformer(
            block_size=block_size,
            stochastic=stochastic,
            use_gzip=use_gzip,
        )
        super().__init__(transformers=[transformer], **kwargs)


Int4BlockPipeline = Int4Pipeline
