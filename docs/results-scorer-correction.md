# The scorer read the model's invented follow-up questions

`extract_number` returns the **last** number in a completion. These models are
base LMs with no stop token for "done answering", so after answering they keep
generating and write their own follow-up exercises. The score then came from
whatever digit that rambling ended on, not from the answer.

A completion that correctly derives `x = 52` and then writes
`Question: Solve for x: 3x + 1 = 10. Answer: x = 3` scored **3**.

Fixed in `evaluate_benchmarks.answer_segment()`: cut the completion at the first
invented `Question:` / `Q:` / numbered exercise, then extract. `Explanation:`
continuations are kept — they belong to the same answer. Tests in
`tests/test_answer_segment.py`.

## Effect on the recorded arms

Re-scoring the stored per-example records:

| Arm | as recorded | corrected | |
|---|---:|---:|---|
| ModernLM pretrain (250M tok) | 95 | **78** | −17 |
| ModernLM pretrain (2B tok) | 115 | **155** | +40 |
| ModernLM 250M + SFT | 163 | 163 | — |
| ModernLM 2B + SFT | 412 | 412 | — |
| 2B + concise SFT | 473 | 473 | — |
| 2B + concise + arithmetic | 497 | 497 | — |
| 2B + concise + arith + number words | 568 | 568 | — |
| 2B + twostep SFT | 361 | 361 | — |
| 2B + words2 SFT | 575 | 575 | — |

**Every SFT arm is unchanged.** SFT supervises `<eos>` and teaches the model to
answer and stop, so it never rambles into a second question and there is nothing
for the fix to cut. The 412 → 568 SFT progression — the main result — stands
exactly as recorded.

Only the pretrain rows move, and they move in both directions: the 2B model was
undercounted (it derives answers and then rambles), the 250M model was
*over*counted (it rambled onto a number that happened to match).

## The token budget interacts with this

The error grows with the generation budget, because a longer completion has more
room to invent questions. Measured on the Muon 2B base:

| budget | as scored | corrected |
|---|---:|---:|
| 32 new tokens | 2.99% | **4.18%** |
| 96 new tokens | 3.58% | **6.61%** |

At 96 tokens the artifact took algebra to **0/100** (12/100 real) and cost asdiv
nearly half its correct answers.

## Which budget to use

Not the same one for every model:

| model | 32 tokens | 96 tokens |
|---|---:|---:|
| 2B base | 4.18% | **6.61%** |
| 2B + SFT | **9.26%** | 8.48% |

A base model works through a problem verbosely and needs the room. An SFT'd
model answers concisely and stops, so extra room only lets it ramble past a
correct answer. **Score base models at 96 and SFT models at 32**, and never
compare across budgets — the difference is 1-2.5pp, the same size as the effects
being measured.
