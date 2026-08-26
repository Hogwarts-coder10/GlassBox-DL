# Known Issues

Local, git-tracked issue log for bugs found during development that aren't
blocking the current review/deadline but need a follow-up fix. Not a
replacement for GitHub Issues — use this for things caught mid-flow that
you want on record before you forget, then optionally promote to a GitHub
issue/PR when you actually pick it up.

Status values: `open` -> `in-progress` -> `fixed` (move fixed entries to
CHANGELOG.md and delete them from here once merged).

---

## OPEN

### ISS-001: `UNetPlusPlus` missing output activation for multi-class
- **Severity:** High (breaks multi-class training if consumer forgets the workaround)
- **Location:** `glassboxdl/models/unetplusplus.py`, `__init__`
- **Description:** `self.out_activation = Sigmoid() if out_classes == 1 else None`
  leaves raw logits for `out_classes > 1`. `CrossEntropyLoss` expects
  softmax'd probabilities per its own docstring, so multi-class output is
  silently wrong unless the caller manually wraps the model output in
  `Softmax(axis=1)`.
- **Workaround in use:** external `Softmax(axis=1)` applied in the training
  script before the loss, for the review-1 demo.
- **Fix:** wire `Softmax(axis=1)` into `out_activation` when `out_classes > 1`,
  same pattern as the existing `Sigmoid()` branch, so the model is correct
  standalone.

### ISS-002: `Conv2D._forward_naive` missing autograd graph link
- **Severity:** Medium (only triggers with `algo="naive"`, not the default)
- **Location:** `glassboxdl/layers/convolution.py`, `_forward_naive`
- **Description:** `out_tensor = Tensor(out, requires_grad=...)` is built
  without `_children=(x, weight, bias)`, so `out_tensor._prev` is empty.
  Autograd's `build_topo` walks the graph via `_prev`, so it never
  recurses past this layer — anything upstream never gets `_backward()`
  called. Same class of bug that was just fixed in `_forward_im2col`.
- **Fix:** mirror the `_forward_im2col` fix — set `_children=(x, self.weight,
  self.bias)` on construction (or `out_tensor._prev = (...)` post-hoc,
  matching the existing style).

### ISS-003: `DiceLoss` internal softmax breaks binary (`out_classes=1`) case
- **Severity:** Medium (silent failure, not a crash)
- **Location:** `glassboxdl/losses/dice.py`, `forward`
- **Description:** `DiceLoss` now unconditionally applies `Softmax(axis=1)`
  to the incoming logits. Correct for multi-class (model leaves logits
  unactivated), but for `out_classes=1` the model already applies
  `Sigmoid()`, and softmax over a single-channel axis always evaluates to
  `1.0` — every pixel prediction collapses to 1.0, loss goes flat, no
  gradient signal, and it fails silently (no error, just doesn't learn).
- **Fix:** guard the internal softmax so it's skipped (or is a no-op) when
  the input has a single channel — e.g. check `logits.shape[1] == 1` — or
  make it an explicit constructor flag (`DiceLoss(apply_softmax=True)`)
  instead of implicit behavior.

---

## FIXED (pending changelog entry)

### ISS-000: naive pooling silently upcast to float64
- **Location:** `glassboxdl/utils/pooling/naive.py`
- **Description:** `maxpool2d_naive`/`avgpool2d_naive` allocated output
  with `np.zeros((N, C, H_out, W_out))` — no dtype, defaulting to
  float64 and upcasting float32 feature maps after the first pool.
  Likely cause of memory fragmentation at 300 samples / 64x64.
- **Fix:** added `dtype=x.dtype` to both allocations.

### ISS-000b: `Conv2D._forward_im2col` missing autograd graph link
- **Location:** `glassboxdl/layers/convolution.py`
- **Description:** same class of bug as ISS-002, but in the default
  `im2col` path — `out_tensor` built without `_children`, breaking
  backprop through every layer before it.
- **Fix:** `out_tensor._prev = (x, self.weight, self.bias)` set after
  construction.
