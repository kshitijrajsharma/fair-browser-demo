import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

HOT_MEAN = np.array([0.4296737853453577, 0.4001659668453235, 0.34333372802741474], dtype=np.float32)
HOT_STD = np.array([0.2056069389373208, 0.16738555558380538, 0.1598986422586595], dtype=np.float32)


def load_tile(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return ((rgb - HOT_MEAN) / HOT_STD).transpose(2, 0, 1)[np.newaxis]


def building_mask(session: ort.InferenceSession, tile: np.ndarray) -> np.ndarray:
    logits = np.asarray(session.run(None, {session.get_inputs()[0].name: tile})[0])
    return 1.0 / (1.0 + np.exp(-logits[0, 0])) >= 0.5


def iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return 1.0 if union == 0 else float(np.logical_and(first, second).sum() / union)


def main(reference_path: Path, candidate_path: Path, tile_paths: list[Path]) -> None:
    tiles = [load_tile(path) for path in tile_paths]
    reference = ort.InferenceSession(reference_path, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(candidate_path, providers=["CPUExecutionProvider"])
    scores = [iou(building_mask(reference, tile), building_mask(candidate, tile)) for tile in tiles]
    print(
        f"{candidate_path.name}: mask IoU vs {reference_path.name} mean {np.mean(scores):.3f} min {np.min(scores):.3f}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sorted(Path(sys.argv[3]).glob("*.png")))
