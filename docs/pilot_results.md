# Pilot Type A JEPA -> Type B alphabet

Date: 2026-06-06

## Data

- Type A train pilot: 10,000 OpenPose sequences, 1,618,031 frames, 17.98 hours.
- Type A validation: 1,739 sequences, 275,233 frames, 3.06 hours.
- Type B alphabet: 670 train, 52 validation, 78 test clips.
- Alphabet splits are signer-disjoint: 7 train, 1 validation, 2 test signers.
- Common representation: 89 joints x 10 features.

The 89 joints contain 8 upper-body joints, 21 joints per hand, and 39 semantic
face points covering eyebrows, eyes, nose, and mouth.

## Hardware and runtime

- GPU: NVIDIA GeForce RTX 2080, 8 GB.
- PyTorch: 2.7.1+cu118.
- JEPA model: 5.30 million parameters, batch 32, 96 frames.
- VRAM during pretraining: about 2.1 GB.
- V3 run: 17,548 steps in 10 minutes.

## Results

All alphabet metrics use unseen signers for validation and test.

| Initialization | Best validation top-1 | Test top-1 | Test top-5 |
| --- | ---: | ---: | ---: |
| Scratch | 51.9% | 62.8% | 91.0% |
| JEPA early, about 3,400 steps | 25.0% | 16.7% | 48.7% |
| JEPA late, about 24,800 steps | 26.9% | 23.1% | 71.8% |
| JEPA v3, 17,548 steps, low fine-tune LR | 15.4% | 7.7% | 37.2% |
| JEPA v3, 17,548 steps, normal fine-tune LR | 11.5% | 5.1% | 24.4% |

Random top-1 chance for 26 letters is 3.85%.

## Interpretation

The end-to-end dataset and training pipeline works, and the scratch alphabet
model is already a useful baseline. Current Type A JEPA representations do not
transfer to the Type B alphabet and should not be used as the default.

V1/V2 showed representation-scale contraction: target embedding norm dropped
from about 11.31 to 8.16 after roughly 24,800 steps. V3 adds point dropout,
variance regularization, norm regularization, and cosine LR decay. It keeps the
norm stable at 11.31, but downstream transfer remains poor.

The most likely remaining causes are:

1. OpenPose and MediaPipe have different missing-point and confidence patterns.
2. Left/right conventions can differ for mirrored MediaPipe clips.
3. The temporal encoder flattens all joints into one token per frame and may
   learn source-specific shortcuts instead of hand-shape structure.
4. Type A general signing and isolated alphabet clips are strongly different
   domains.

## Next training work

1. Make left/right reflection augmentation swap body and hand identities.
2. Replace frame flattening with spatial joint tokens or a graph transformer.
3. Remove confidence as a direct input and use validity only as an attention mask.
4. Add a same-domain self-supervised control on Type B train clips.
5. Evaluate frozen linear probes before full fine-tuning.
6. Keep the scratch 62.8% model as the baseline to beat.
