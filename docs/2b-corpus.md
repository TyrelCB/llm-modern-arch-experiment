# The 2B-token corpus

Built for the 2B run because the original packed corpus (260M tokens) would
have meant 7.69 epochs at this budget, and a 145M model repeating data that
many times memorizes rather than generalizes.

## Provenance

| Item | Value |
|---|---:|
| Source | HuggingFaceTB/finemath, `finemath-4plus` (ODC-By 1.0) |
| Shards | 32 of 64 |
| Training documents | 1,374,969 |
| Packed training tokens | 2,050,000,000 |
| Packed held-out tokens | 5,000,000 |
| Contaminated documents rejected | 759 |
| Split seed | `deepseek-v4-finemath-v1` (unchanged) |
| Held-out permille | 10 (unchanged) |
| Decontamination | 13-gram vs GSM8K/SVAMP/ASDiv (unchanged) |

## What was deliberately reused, not rebuilt

**The tokenizer is loaded from the reference's `tokenizer.json`, byte-identical
(sha256 verified).** Retraining BPE on a larger sample would shift every token
id and change sequence lengths, making held-out loss incomparable to both the
250M baseline and the DeepSeek-V4 reference.

The split seed and holdout rule are also unchanged, which is what makes the
expansion safe. Verified after packing:

| Check | Result |
|---|---:|
| new train ∩ new held-out | 0 |
| new held-out ∩ original train | 0 |
| new train ∩ original held-out | 0 |

The third row is the important one: because the split is a hash of the document
key rather than a position in a list, a document held out at 260M is still held
out at 2B. The 2B model never trains on the 250M run's evaluation data.

## Reproduce

```bash
python scripts/prepare_2b_corpus.py
```
