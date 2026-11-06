# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Test module for Int4Pipeline and Int4Transformer."""

import numpy as np
import pytest

from openfl.pipelines import Int4BlockPipeline, Int4Pipeline, Int4Transformer, TensorCodec
from openfl.protocols import base_pb2
from openfl.utilities import TensorKey


class TestInt4Pipeline:
    """Test suite for Int4Pipeline and Int4Transformer."""

    def test_invalid_parameters(self):
        """Test that invalid constructor parameters raise ValueError."""
        with pytest.raises(ValueError):
            Int4Transformer(block_size=0)
        with pytest.raises(ValueError):
            Int4Transformer(block_size=-16)
        with pytest.raises(ValueError):
            Int4Transformer(block_size=64.0)
        with pytest.raises(ValueError):
            Int4Transformer(block_size=True)
        with pytest.raises(ValueError):
            Int4Transformer(stochastic="yes")
        with pytest.raises(ValueError):
            Int4Transformer(use_gzip=1)

    def test_pipeline_is_lossy(self):
        """Test that Int4Pipeline reports itself as lossy."""
        pipeline = Int4Pipeline(block_size=64)
        assert pipeline.is_lossy() is True
        assert Int4BlockPipeline is Int4Pipeline

    def test_blockwise_quantization_roundtrip(self):
        """Test block-wise INT4 quantization and dequantization fidelity."""
        rng = np.random.default_rng(42)
        weights = rng.standard_normal((10, 20)).astype(np.float32)

        pipeline = Int4Pipeline(block_size=32)
        data_bytes, metadata = pipeline.forward(weights)

        assert isinstance(data_bytes, bytes)
        assert len(metadata) == 1
        # 200 float32s = 800 bytes; padded to 7x32=224 elements -> 112 packed bytes
        assert len(data_bytes) == 112
        assert len(data_bytes) < weights.nbytes // 6

        recovered = pipeline.backward(data_bytes, metadata)
        assert recovered.shape == weights.shape
        assert recovered.dtype == np.float32

        mse = np.mean((weights - recovered) ** 2)
        assert mse < 0.05

    def test_whole_tensor_quantization(self):
        """Test whole-tensor dynamic scaling when block_size is None."""
        rng = np.random.default_rng(7)
        weights = rng.standard_normal((16, 16)).astype(np.float32)

        pipeline = Int4Pipeline(block_size=None)
        data_bytes, metadata = pipeline.forward(weights)

        assert isinstance(data_bytes, bytes)
        assert len(metadata[0]["int_to_float"]) == 1

        recovered = pipeline.backward(data_bytes, metadata)
        assert recovered.shape == weights.shape
        mse = np.mean((weights - recovered) ** 2)
        assert mse < 0.05

    def test_with_gzip_compression(self):
        """Test that use_gzip=True compresses packed bytes and reconstructs correctly."""
        rng = np.random.default_rng(99)
        weights = rng.standard_normal((32, 32)).astype(np.float32)

        pipeline = Int4Pipeline(block_size=64, use_gzip=True)
        data_bytes, metadata = pipeline.forward(weights)

        assert isinstance(data_bytes, bytes)
        recovered = pipeline.backward(data_bytes, metadata)
        assert recovered.shape == weights.shape

        mse = np.mean((weights - recovered) ** 2)
        assert mse < 0.05

    def test_stochastic_rounding_bounds(self):
        """Test stochastic rounding quantization."""
        rng = np.random.default_rng(123)
        weights = rng.standard_normal((8, 8)).astype(np.float32)

        pipeline = Int4Pipeline(block_size=16, stochastic=True)
        data_bytes, metadata = pipeline.forward(weights)
        recovered = pipeline.backward(data_bytes, metadata)

        assert recovered.shape == weights.shape
        diff = np.abs(weights - recovered)
        assert np.all(diff < 2.0)

    def test_all_zeros_handling(self):
        """Test that all-zero weight matrices do not produce divide-by-zero or NaNs."""
        zeros = np.zeros((12, 12), dtype=np.float32)
        pipeline = Int4Pipeline(block_size=16)

        data_bytes, metadata = pipeline.forward(zeros)
        recovered = pipeline.backward(data_bytes, metadata)

        np.testing.assert_array_equal(recovered, zeros)

    def test_non_finite_rejection(self):
        """Test that NaN or Inf inputs are rejected immediately."""
        pipeline = Int4Pipeline()
        with_nan = np.array([1.0, np.nan, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Input array contains non-finite values"):
            pipeline.forward(with_nan)

        with_inf = np.array([1.0, np.inf, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Input array contains non-finite values"):
            pipeline.forward(with_inf)

    def test_empty_array_handling(self):
        """Test handling of 0-element arrays."""
        empty = np.array([], dtype=np.float32)
        pipeline = Int4Pipeline()

        data_bytes, metadata = pipeline.forward(empty)
        recovered = pipeline.backward(data_bytes, metadata)
        assert recovered.size == 0

    def test_unaligned_and_odd_size(self):
        """Test tensor with odd number of elements not aligned with block_size."""
        rng = np.random.default_rng(55)
        arr = rng.standard_normal(37).astype(np.float32)

        pipeline = Int4Pipeline(block_size=16)
        data_bytes, metadata = pipeline.forward(arr)
        recovered = pipeline.backward(data_bytes, metadata)

        assert recovered.shape == arr.shape
        np.testing.assert_allclose(recovered, arr, atol=0.5)

    def test_tensor_codec_integration(self):
        """Test integration with OpenFL's TensorCodec."""
        rng = np.random.default_rng(2024)
        weights = rng.standard_normal((10, 10)).astype(np.float32)

        codec = TensorCodec(Int4Pipeline(block_size=32))
        tensor_key = TensorKey("conv1.weight", "client_1", 1, False, ("trained",))

        comp_key, comp_bytes, meta = codec.compress(tensor_key, weights)
        assert "lossy_compressed" in comp_key[4]
        assert isinstance(comp_bytes, bytes)

        decomp_key, decomp_weights = codec.decompress(comp_key, comp_bytes, meta)
        assert "lossy_decompressed" in decomp_key[4]
        assert decomp_weights.shape == weights.shape
        assert np.mean((weights - decomp_weights) ** 2) < 0.05

    def test_protobuf_metadata_wire_compatibility(self):
        """Test that transformer metadata serializes into OpenFL MetadataProto without errors."""
        rng = np.random.default_rng(333)
        weights = rng.standard_normal((8, 16)).astype(np.float32)

        pipeline = Int4Pipeline(block_size=32)
        data_bytes, metadata = pipeline.forward(weights)

        proto = base_pb2.MetadataProto()
        proto.int_to_float.update(metadata[0]["int_to_float"])
        proto.int_list.extend(metadata[0]["int_list"])
        proto.bool_list.extend(metadata[0]["bool_list"])

        serialized = proto.SerializeToString()
        assert len(serialized) > 0

        deserialized = base_pb2.MetadataProto()
        deserialized.ParseFromString(serialized)

        deserialized_meta = [
            {
                "int_to_float": dict(deserialized.int_to_float),
                "int_list": list(deserialized.int_list),
                "bool_list": list(deserialized.bool_list),
            }
        ]

        recovered = pipeline.backward(data_bytes, deserialized_meta)
        assert recovered.shape == weights.shape
        np.testing.assert_allclose(recovered, weights, atol=0.5)
