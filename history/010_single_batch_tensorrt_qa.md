# 010: Single-batch TensorRT policy for synthetic ONNX QA

Created 2026-08-31 (user request: prefer single-batch TensorRT and never use multiple-batch ONNX inference)

## 0. Decision

Every synthetic-QA ONNX invocation is now batch size 1. SixD and HRFFA provider priority is TensorRT, CUDA,
then CPU. DEIMv2 is deliberately restricted to CUDA, then CPU, because TensorRT inference has an unexplained
accuracy regression. The existing `--cpu` flag remains an explicit CPU override. Multiple-batch inference is
forbidden because it is slower for this workload and can make TensorRT engine-cache behavior unstable.

The public model inputs were inspected with ONNX Runtime:

| model | public input |
|---|---|
| DEIMv2 boxes-only | `images [N,3,640,640]` |
| SixDRepNet360 | `input [1,3,224,224]` |
| HRFFA ViT-L iBUG68 | `images [1,3,320,320]` |

DEIM retains its upstream symbolic-`N` graph, but the wrapper converts each image independently to
`[1,3,640,640]` and calls `session.run` once per image. Its compatibility `infer_batch` API is sequential and
never stacks records. SixD and HRFFA reject a model whose public batch dimension is not fixed to 1, and all
three wrappers check the actual inference tensor immediately before `session.run`.

The `optimize-onnx-batches` skill was consulted because DEIM exposes symbolic `N`. No ONNX graph rewrite was
performed: this change requires a runtime batch-1 invariant, not a separately published fixed/dynamic model
pair. Avoiding a derived DEIM artifact also preserves the already hash-pinned upstream model. If a fixed DEIM
artifact is requested later, it must follow the skill's fixed-first checker, simplification, parity, and atomic
publication requirements.

## 1. TensorRT cache isolation

When `TensorrtExecutionProvider` is available, its engine and timing caches use:

```text
data/models/trt_cache/ort-<ORT>_trt-<TensorRT>_cuda-<CUDA>_sm<CC>_<precision>/
  <model stem>-<SHA-256 prefix>-batch1/
```

For SixD and HRFFA, the path separates ONNX Runtime, TensorRT, CUDA runtime, GPU compute capability, precision, model revision, and
the only permitted batch profile. A change in any runtime field selects a different empty directory, making
cross-runtime cache reuse impossible. Runtime identification is fail-closed: TensorRT execution stops rather
than use an unversioned cache if its runtime fingerprint cannot be read. Each model has its own directory.
CUDA and CPU remain fallback providers after TensorRT. `qa_report.json` records the requested priority,
selected provider, actual session providers, full runtime fingerprint, cache directory, and the batch-1
invariant for SixD and HRFFA. DEIM reports its CUDA/CPU provider plan and has no TensorRT cache.

## 2. Verification

- Provider-policy tests simulate TensorRT+CUDA availability and verify TensorRT is first and the cache path is
  model-hash/batch-1 scoped.
- A DEIM regression test passes three images to the compatibility API and verifies three separate
  `(1,3,640,640)` executions.
- Real DEIM, SixD, and HRFFA CPU-fallback inference succeeded. The current environment exposes Azure and CPU
  ONNX Runtime providers only, so actual TensorRT engine construction cannot be exercised here.
- `git diff --check` and Python compilation passed; `uv run --locked pytest -q` passed 69 tests with four
  pre-existing PyTorch ONNX-export deprecation warnings.

## 3. Runtime replacement verification

After `onnxruntime-gpu 1.26.0` was installed, the available providers were TensorRT, CUDA, and CPU. The runtime
fingerprint was `ORT 1.26.0 / TensorRT 10.14.1 / CUDA 13.0 / SM86 / FP32`. The earlier ORT-only cache directory
was moved to the desktop trash and was not reused. Fresh batch-1 caches were built for SixD, DEIM, and HRFFA
under the complete fingerprint path. Each first build reported that no timing cache could be read and created a
new one; a second process then loaded the new caches successfully. After the DEIM TensorRT accuracy issue was
specified, its TensorRT cache was retired and the runtime policy was changed to CUDA-only with CPU fallback.
