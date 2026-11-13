import sys
from pathlib import Path

from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

FLOAT16_PATCH_EMBEDDING_CONV = "node_Conv_3"


def quantize(source: Path, target: Path) -> None:
    quantize_dynamic(
        source,
        target,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=[FLOAT16_PATCH_EMBEDDING_CONV],
    )
    print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    source_path = Path(sys.argv[1])
    print(f"{source_path.name}: {source_path.stat().st_size / 1e6:.1f} MB")
    preprocessed_path = source_path.with_name(source_path.stem + "_preprocessed.onnx")
    quant_pre_process(source_path, preprocessed_path)
    print(f"{preprocessed_path.name}: {preprocessed_path.stat().st_size / 1e6:.1f} MB")
    quantize(preprocessed_path, source_path.with_name(source_path.stem + "_uint8.onnx"))
