# Toy Supervised Fine-Tuning from Scratch

This repository is a minimal educational implementation of supervised fine-tuning (SFT) with PyTorch and Hugging Face Transformers. It trains a local `Qwen3-0.6B-Base` checkpoint on 40 short addition questions, then compares that checkpoint with the untouched base model on 20 held-out questions.

The purpose is to learn the mechanics. The dataset is intentionally small. The training loop is ordinary next-token cross-entropy on the answer tokens, written out step by step: tokenization, label masking, padding, a `DataLoader`, a forward pass, backpropagation, and an AdamW update. This is not a production training framework, and the toy data is not a demonstration of state-of-the-art performance.

## What you will learn

- What SFT changes relative to pretraining: the examples, and which token positions count in the loss.
- How a question and an answer become one `input_ids` sequence.
- Why prompt tokens use the label `-100` while answer tokens keep their ids.
- Why padding needs an attention mask of `0` and a label of `-100`.
- What one optimization step does: forward, loss, `backward`, `optimizer.step()`.
- Why a falling training loss is not the same thing as a better generated answer.
- How to compare a base checkpoint and a fine-tuned checkpoint with the same prompt and the same greedy decoder.

## SFT workflow

```text
Pretrained Model
       |
       v
Prompt-Completion Dataset
       |
       v
Tokenization and Label Preparation
       |
       v
Batching
       |
       v
Forward Pass
       |
       v
Loss Calculation
       |
       v
Backpropagation
       |
       v
Optimizer Update
       |
       v
Fine-Tuned Model
       |
       v
Evaluation and Comparison
```

## Project structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   └── 01_supervised_fine_tuning.ipynb
├── data/
│   ├── train.json          # 40 questions used for gradient updates
│   └── heldout.json        # 20 questions used only for evaluation
└── models/
    └── .gitkeep            # local checkpoints live here and are not committed
```

| Path | Purpose |
|---|---|
| `notebooks/01_supervised_fine_tuning.ipynb` | The tutorial. Run the cells from top to bottom. |
| `data/train.json` | The only examples whose tokens enter the loss. |
| `data/heldout.json` | Unseen number pairs used for loss and generation after training. |
| `models/Qwen3-0.6B-Base` | The pretrained checkpoint you provide. The notebook never overwrites it. |
| `models/Qwen3-0.6B-SFT` | Written by the notebook after training. Used for the comparison. |

## Requirements

- Python 3.10 or newer.
- The packages in `requirements.txt`. This lesson was developed with `torch` 2.14.0 and `transformers` 5.17.0.
- A local copy of `Qwen3-0.6B-Base`, about 1.2 GB. The notebook does not download weights. The checkpoint's license is separate from the MIT license on this tutorial code.

## Setup

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-0.6B-Base', local_dir='models/Qwen3-0.6B-Base')"
jupyter notebook notebooks/01_supervised_fine_tuning.ipynb
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen3-0.6B-Base', local_dir='models/Qwen3-0.6B-Base')"
jupyter notebook notebooks/01_supervised_fine_tuning.ipynb
```

The download writes the checkpoint into `models/Qwen3-0.6B-Base`. These files need to be there before the notebook will load the model:

```text
models/Qwen3-0.6B-Base/config.json
models/Qwen3-0.6B-Base/model.safetensors
models/Qwen3-0.6B-Base/tokenizer.json
```

Run the notebook from the first cell downward. It expects the working directory to be the repository root or `notebooks/`. Training uses CUDA if it is available, otherwise Apple MPS, otherwise CPU. The save cell writes `models/Qwen3-0.6B-SFT` and replaces that folder if it already exists.

The training loader shuffles with seed 0, so the batch order stays fixed. The numbers quoted below come from one seeded run. Another device can still print different decimals. To follow the device used for those numbers, uncomment `device = torch.device("cpu")` in the model-loading cell only if that run was on CPU.

## Dataset format

Both JSON files are lists of records:

```json
[
  {
    "question": "What is 2 + 3?",
    "answer": "Answer: 5"
  }
]
```

The training file has 40 rows. The held-out file has 20 rows. No held-out operand pair appears in training. Held-out rows are never passed to the optimizer. The target string includes the prefix `Answer:`, so the supervised behaviour is both the sum and that format.

The prompt built for every row is:

```text
Question: What is 2 + 3?
```

The supervised continuation is `Answer: 5` plus the tokenizer's end-of-sequence token.

This is one task in prompt-completion form. The question is the whole prompt. There is no separate instruction field, and the file is not a multi-task instruction mixture.

### One training row, after tokenization

The first row of `data/train.json` is `What is 0 + 0?` with target `Answer: 0`. The notebook turns it into this string:

```text
Question: What is 0 + 0?
Answer: 0<|endoftext|>
```

The Qwen3-0.6B-Base tokenizer, with `add_special_tokens=False`, splits that string as follows. A leading `Ġ` is a space. The `Ċ` inside `?Ċ` is the newline at the end of the question. `<|endoftext|>` is the end token, id `151643`.

| i | token | input_id | attention | label |
|---|---|---|---|---|
| 0 | Question | 14582 | 1 | -100 |
| 1 | : | 25 | 1 | -100 |
| 2 | ĠWhat | 3555 | 1 | -100 |
| 3 | Ġis | 374 | 1 | -100 |
| 4 | Ġ | 220 | 1 | -100 |
| 5 | 0 | 15 | 1 | -100 |
| 6 | Ġ+ | 488 | 1 | -100 |
| 7 | Ġ | 220 | 1 | -100 |
| 8 | 0 | 15 | 1 | -100 |
| 9 | ?Ċ | 5267 | 1 | -100 |
| 10 | Answer | 16141 | 1 | 16141 |
| 11 | : | 25 | 1 | 25 |
| 12 | Ġ | 220 | 1 | 220 |
| 13 | 0 | 15 | 1 | 15 |
| 14 | `<|endoftext|>` | 151643 | 1 | 151643 |

The first 10 positions are the question. They stay in `input_ids` so the model can read them, and their labels are `-100`, so they do not enter the loss. The last 5 positions are the answer and the end token. Their labels are the token ids themselves. This row has no padding. Padding is added later, only so shorter rows can sit in the same tensor.

Hugging Face compares the logit at each position with the label at the next position. The logit on the last question token, `?Ċ`, is scored against the first answer label, `Answer`. The question stays in the forward pass as context. `-100` only removes the question's own next-token terms from the loss.

The same row with no mask trains a different thing. Every label is now the token id itself:

| i | token | input_id | attention | label |
|---|---|---|---|---|
| 0 | Question | 14582 | 1 | 14582 |
| 1 | : | 25 | 1 | 25 |
| 2 | ĠWhat | 3555 | 1 | 3555 |
| 3 | Ġis | 374 | 1 | 374 |
| 4 | Ġ | 220 | 1 | 220 |
| 5 | 0 | 15 | 1 | 15 |
| 6 | Ġ+ | 488 | 1 | 488 |
| 7 | Ġ | 220 | 1 | 220 |
| 8 | 0 | 15 | 1 | 15 |
| 9 | ?Ċ | 5267 | 1 | 5267 |
| 10 | Answer | 16141 | 1 | 16141 |
| 11 | : | 25 | 1 | 25 |
| 12 | Ġ | 220 | 1 | 220 |
| 13 | 0 | 15 | 1 | 15 |
| 14 | `<|endoftext|>` | 151643 | 1 | 151643 |

That version asks the model to reproduce the question as well as the answer. The question is already on the page at inference, so those extra terms teach copying. The `-100` mask is what keeps the loss on the answer.

### Use your own file

Keep the same two fields, `question` and `answer`, and the same list shape. Put training rows in `data/train.json` and rows that must not update the weights in `data/heldout.json`. The notebook always builds the prompt as `Question: ` plus the question plus a newline, and it always appends the end token to `answer`. If your target should not start with the words `Answer: `, change the text in the JSON. The mask follows that split. It does not look for the word `Answer`.

If your records use different field names, change `make_example` in the notebook. Leave held-out rows out of the training file.

## Implementation walkthrough

The notebook is the full implementation. The update rule matches the run that produced the results below: the same prompt, the same `-100` mask, global padding, batch size 4, AdamW at `5e-5`, and five epochs. Paths are relative to the repository. Training and held-out rows share one `make_example` function. The model is moved to a selected device. Both generation checks use `max_new_tokens=32`, which is what the recorded side-by-side comparison used.

**Tokenization.** The question and the answer are encoded separately with `add_special_tokens=False`, then concatenated. Encoding them as one string and slicing at a character boundary can merge tokens across the cut.

**Label masking.** Prompt positions are labeled `-100`. Answer positions, including the end token, keep their token ids. `-100` is the ignore index, so those positions stay in the forward pass and drop out of the loss. Hugging Face shifts labels inside the model. This code does not shift them again.

**Padding.** `pad_sequence` stretches every training row to the longest row in the file. Padded inputs use the pad id, padded attention uses `0`, and padded labels use `-100`. The pad id is the end-of-sequence id, because this tokenizer has no separate pad token. The `-100` labels are what stop that reuse from becoming an extra training target.

**DataLoader.** `TensorDataset` holds the three padded tensors. The training loader uses batch size 4, `shuffle=True`, and seed 0, which is 10 optimizer steps per epoch. The held-out loader uses the same batch size and `shuffle=False`.

**Forward pass.** `AutoModelForCausalLM` reads `input_ids` and `attention_mask`. Passing `labels` makes it return the masked next-token loss on `outputs.loss`.

**Loss.** The loss is the mean negative log probability of the supervised tokens in the batch, after the library's internal shift. The epoch printout is the unweighted mean of those batch losses.

**Backpropagation.** `loss.backward()` writes a gradient for each trainable parameter. It does not itself change the weights.

**AdamW update.** `optimizer.zero_grad()` clears the previous batch's gradients, and `optimizer.step()` applies AdamW at learning rate `5e-5`. Every pretrained parameter is trainable. `5e-5` and five epochs are the settings of this toy run, not a general recipe.

**Evaluation.** Held-out loss reuses the same masking and runs under `model.eval()` and `torch.no_grad()`. No optimizer step runs there. Generation then feeds only the question prompt and decodes greedily with `max_new_tokens=32`.

**Model comparison.** The base folder and the saved fine-tuned folder are loaded separately. Both see the same prompt and the same decoder. Accuracy is exact string match against the `answer` field.

## Results

These numbers are from the earlier CPU run of this loop, before shuffle was pinned to seed 0. They are not a general claim about SFT. A new run uses seed 0, so the batch order is fixed. On another device the decimals can still change.

Settings for that earlier run: `Qwen3-0.6B-Base`, AdamW at `5e-5`, batch size 4, five epochs, model left on CPU, greedy decoding, exact match on the answer string.

| Epoch | Average training loss |
|---|---|
| 1 | 1.2441 |
| 2 | 0.0804 |
| 3 | 0.0153 |
| 4 | 0.0034 |
| 5 | 0.0023 |

Held-out loss after that run was **0.0361**.

On the 20 held-out questions, exact-match accuracy was **100% for the base model (20/20)** and **95% for the fine-tuned model (19/20)**. The only difference was `What is 1 + 18?`: the expected text is `Answer: 19`, the base model produced `Answer: 19`, and the fine-tuned model produced `Answer: 29`.

Training loss fell because the model became better at imitating the 40 training answers under teacher forcing. On this held-out set, the update did not improve exact-match accuracy. The base checkpoint already answered these prompts correctly, and fine-tuning changed one of those answers to a wrong sum.

## Limitations

- Forty training rows and twenty test rows are enough to see the tensors. They are not enough to conclude anything about addition in general.
- The task is narrow: one prompt template and integer sums the base model already handles.
- Exact match treats a format change as a failure even if the number is right, and it treats a wrong number in the right format as a failure. Both are intended here, and both are blunt.
- Padding is global rather than per batch, and the learning rate stays constant. The shuffle seed is fixed at 0. That fixes the batch order, not the decimals on every device.
- The recorded comparison is one seeded run. It is not a multi-seed study.
- Full fine-tuning updates every weight. There is no adapter, gradient checkpointing, or mixed-precision strategy beyond the checkpoint's own dtype.

## Learning outcomes

After the notebook, you should be able to point at a training row and say which tokens are context, which tokens create loss, and which positions are padding. You should be able to name the four calls that make one update: forward, read `outputs.loss`, `backward`, `step`. You should also be able to explain why the loss printed during training can fall while the base model remains the better generator on the held-out questions.

## Future extensions

These are optional next steps. The basic loop above does not depend on them.

- Train a LoRA adapter instead of every weight, and compare it with this full update on the same files.
- Pad each batch to its own longest row instead of padding the whole dataset first.
- Save a checkpoint every epoch and choose one with held-out generation, rather than keeping the last epoch automatically.
- Replace the addition file with a larger instruction set, and score a behaviour the base model does not already solve.
- Add a chat template only when both training and evaluation use that same template.

## License

The tutorial code is released under the MIT license in `LICENSE`. The Qwen checkpoint is not covered by that license. Follow the license shipped with the model weights you place in `models/`.
