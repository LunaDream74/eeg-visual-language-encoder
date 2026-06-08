  1. Main scatter (left, full-height) — "the big picture"

  Shows all 30 concept-points in 2D, using synthetic embeddings projected down to 2 axes (as PCA would do with real
  768-dim embeddings). Blue circles = CLIP image embeddings (the fixed, well-structured target space). Orange circles =
  EEG embeddings (what the brain encoder is predicting). White lines connect every matched pair.

  - Why it looks like this before training: the encoder has random weights, so EEG outputs land anywhere in space — the
  lines are long and tangled, showing total misalignment.
  - Why it animates the way it does during convergence: each frame advances interpolation parameter t from 0→1 with a
  smooth-step curve (t²(3-2t)), so early steps are chaotic (jitter is high), then the orange dots accelerate toward
  their blue targets, mimicking the fast early learning then slower fine-tuning seen in real training.
  - Why semantic colours appear at the end: after training, CLIP's own structure (animals cluster together, vehicles
  cluster together) is preserved in the aligned space — the EEG encoder has learned not just to match individual pairs
  but to respect the whole semantic geometry.

  ---
  2. Batch highlight (top-middle) — "what the loss sees right now"

  The same 30 points, but all non-batch members are faded to 25% opacity. The 8 selected pairs are ringed in yellow with
   yellow connector lines.

  - Why only 8 are highlighted: InfoNCE is a mini-batch loss. At each gradient step the model sees exactly N pairs. All
  N² cross-pair similarities are computed, but the loss function has no knowledge of the other 22 concepts — they do not appear in the InfoNCE numerator or denominator. However, because the EEG encoder has shared weights, the gradient from these 8 pairs updates the encoder for all inputs. The animation shows this aggregate effect: all 30 points drift over many batches, even though only 8 are active in any one loss computation.
  This panel shows that the "rubber band" metaphor applies specifically to the current batch — not to all 30 points at once.
  - Why the connectors get shorter over time: as training progresses (phase 2), eeg_cur is re-computed each frame with a
   higher t, so the batch pairs naturally move closer, and the yellow lines physically shorten.

  ---
  3. Similarity matrix (top-right) — "the N×N score card"

  An 8×8 heatmap. Row i = EEG embedding i from the batch. Column j = CLIP embedding j from the batch. Cell (i,j) =
  cosine similarity between them. Red = +1, blue = −1, white = 0. Yellow boxes outline the diagonal (the 8 matched
  pairs).

  - Why the diagonal should be red: (i,i) is the matched pair — same stimulus, so the model should output similar
  vectors for brain and image. InfoNCE's numerator is exactly exp(sim(i,i)/τ).
  - Why off-diagonal should be blue: (i,j≠i) are mismatches — unrelated brain signal and image. InfoNCE's denominator
  sums over all j, so keeping off-diagonal low makes the fraction large, driving the loss down.
  - Why it looks noisy before training and clean after: noise σ=0.25·(1-t) is added to the similarity values, reflecting
   that before training the dot products are essentially random. As t→1 the noise drops to zero and the structure
  appears.

  ---
  4. Pull/push arrows (bottom-middle) — "the forces InfoNCE applies"

  The same 8 batch points, now with explicit force vectors drawn from each EEG dot.

  - Green arrow (→ matched CLIP dot): the gradient of the InfoNCE loss with respect to the EEG embedding points toward
  the matched CLIP embedding. This is the "pull" — the loss is minimised when these two vectors are identical.
  - Red arrow (→ away from a mismatched CLIP dot): the negative gradient contribution from off-diagonal pairs pushes the
   EEG embedding away from wrong answers. The arrow points in the direction opposite to the mismatched CLIP dot
  (computed as (eeg_pos - clip_neg) / ‖…‖ × scale).
  - Why both forces are always visible regardless of phase: the pull/push logic is the same mathematical operation every
   step — what changes is the magnitude and the starting positions of the dots as training progresses.

  ---
  5. Loss curve (bottom-right) — "the quantitative signal"

  A line chart of the InfoNCE loss value at each of the 96 convergence steps, drawn progressively (one more point per
  frame in phase 2).

  - Why it starts high (~3.4): with random embeddings, all N dot products are near zero and the softmax is nearly
  uniform — the loss approaches log(N) ≈ log(8) ≈ 2.1, plus initial noise. The model has no information yet.
  - Why it falls exponentially then flattens: the synthetic curve uses L(t) = 3e^{-3.5t} + 0.4 + noise, mimicking the
  empirical loss trajectory seen in real training (fast initial descent as the encoder learns rough alignment, slow
  plateau as it fine-tunes).
  - Why an orange dot tracks the current step: it gives a visual "you are here" marker so the viewer can correlate the
  scatter plot's state of alignment with the exact point on the loss curve.