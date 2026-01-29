from distillkit.compression.bitpack import (
    pack_to_bytes,
    unpack_from_bytes,
)
from distillkit.compression.config import (
    DistributionQuantizationConfig,
    LegacyLogitCompressionConfig,
    QuantizationBin,
)

__all__ = [
    "pack_to_bytes",
    "unpack_from_bytes",
    "QuantizationBin",
    "DistributionQuantizationConfig",
    "LogprobCompressor",
    "densify",
    "LegacyLogitCompressor",
    "LegacyLogitCompressionConfig",
]


def __getattr__(name: str):
    # Lazy imports to avoid circular import during configuration load.
    if name == "LogprobCompressor":
        from distillkit.compression.compressor import LogprobCompressor

        return LogprobCompressor
    if name == "densify":
        from distillkit.compression.densify import densify

        return densify
    if name == "LegacyLogitCompressor":
        from distillkit.compression.legacy import LogitCompressor as LegacyLogitCompressor

        return LegacyLogitCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
