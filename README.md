# fAIr building model in the browser

Static page that runs the fAIr building model [`dinov3s_buildings`](https://github.com/hotosm/fAIr-models/tree/develop/models/dinov3s_buildings) on the viewer's device with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.31.1 and reports download and inference time. Paste an imagery.hotosm.org link, OpenAerialMap image ID or STAC item URL to step through that image's zoom 19 tiles.

## Models

Both files are on https://huggingface.co/kshitijrajsharma/dinov3-hot-buildings.

| File | Size | Mask IoU vs 32-bit |
|---|---|---|
| `dinov3s_buildings.onnx` | 219.6 MB | reference |
| `dinov3s_buildings_uint8.onnx` | 76.6 MB | 0.991 mean, 0.987 min over 9 Mtandire tiles |

The 8-bit file comes from ONNX Runtime [dynamic quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html):

1. `quant_pre_process` folds the encoder's float32-to-float16 Cast nodes into float16 weights (219.6 to 163.2 MB).
2. `quantize_dynamic` stores the 13 decoder convolutions as uint8 (163.2 to 76.6 MB). The float16 encoder stays as it is.

## Reproduce

Download `dinov3s_buildings.onnx` into `models/`, then:

```bash
just setup
uv run python quantize.py models/dinov3s_buildings.onnx
uv run python compare.py models/dinov3s_buildings.onnx models/dinov3s_buildings_uint8.onnx samples
```

Serve the folder with `python3 -m http.server` and open `/?models=./models/` to load the local files.

## Notes

- The page keeps downloaded models in browser [Cache Storage](https://developer.mozilla.org/en-US/docs/Web/API/CacheStorage), because Hugging Face answers model requests with a `cache-control: no-store` redirect to a signed URL.
- WASM runs multi-threaded only on a [cross-origin isolated](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated) page. GitHub Pages cannot send the required headers ([discussion](https://github.com/orgs/community/discussions/13309)), so it runs on one thread there.

## Licences

| Part | Licence |
|---|---|
| Model weights | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), per the fAIr-models [STAC item](https://github.com/hotosm/fAIr-models/blob/develop/models/dinov3s_buildings/stac-item.json) |
| fAIr-models code | [AGPL-3.0](https://github.com/hotosm/fAIr-models/blob/develop/LICENSE) |
| `samples/` tiles | CC-BY-SA-4.0, from [Mtandire, corrected 2026-10-03](https://imagery.hotosm.org/browse?selected_id=00d19bd9-3222-47d8-9a6f-9f39339ac487&lat=-13.9492&lon=33.7376&zoom=15) on OpenAerialMap |


slide : https://docs.google.com/presentation/d/15pnOwhdifOyr6wdaraJqjxzzbT89gHjwRca-6LCZmZw/edit?usp=sharing
