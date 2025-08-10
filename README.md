## GPT-2 Fine‑tuning and From‑Scratch Notebook

This project contains an educational, end‑to‑end workflow to build and fine‑tune a GPT‑style language model using PyTorch, with all steps implemented in the notebook `gpt2.ipynb`. It also includes utilities to prepare the MedQuAD dataset into Alpaca format and optional exploration of the Hugging Face `nvidia/HelpSteer` dataset.

### Highlights
- From‑scratch GPT mini‑architecture in PyTorch (embeddings, positional encodings, masked multi‑head attention, feed‑forward blocks, residuals, layer norm)
- Tokenization via `tiktoken` using GPT‑2 vocabulary
- Data loaders for contiguous text and instruction‑style examples (Alpaca format)
- Simple training loop with evaluation, sample generation, and checkpoint saving
- Dataset prep helpers for MedQuAD → Alpaca JSONL; example loading of `nvidia/HelpSteer`

## Project structure
- `gpt2.ipynb`: Primary notebook with the end‑to‑end code (tokenization, model, dataloaders, training, generation, dataset prep)
- `gpt2/355M/`: Reference GPT‑2 assets (original OpenAI 355M files). Not used by default in the notebook
- `last_model.pt`, `gpt2_small_trained.pth`, `fine_tuned.pth`: Saved model weights from training runs
- `medquad.csv`: MedQuAD dataset CSV (input)
- `medquad_alpaca.jsonl`: MedQuAD converted to Alpaca JSONL (produced by the notebook)

## Requirements
- Python 3.10+
- Recommended: NVIDIA GPU with CUDA 11.8 (or CPU, slower)

Python packages used in the notebook:
- `torch`, `torchvision`, `torchaudio` (CUDA 11.8 build shown in examples)
- `tiktoken`, `datasets`, `pandas`, `matplotlib`, `ipywidgets`
- Optional: `transformers`, `huggingface_hub`, `tensorflow` (used only for exploration/visuals)

Quick install (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken datasets pandas matplotlib ipywidgets transformers huggingface_hub
```

## Getting started
1) Open `gpt2.ipynb` in Jupyter (or VS Code/Colab). Ensure the environment has the packages above.
2) Run the notebook cells top‑to‑bottom. Early cells demonstrate tokenization and the model components; later cells assemble the training loop and generation utilities.

## Data preparation
The notebook contains cells to:
- Load `nvidia/HelpSteer` via `datasets.load_dataset("nvidia/HelpSteer")` for inspection/examples
- Convert `medquad.csv` into Alpaca JSONL (`medquad_alpaca.jsonl`) with fields: `instruction`, `input`, `output`
- Optionally clean, filter by token length, and split into train/val/test JSONL files

Inputs/outputs:
- Input: `medquad.csv`
- Output: `medquad_alpaca.jsonl` and optional split files (created when running the relevant cells)

## Training
Training is implemented inside `gpt2.ipynb` with helpers such as:
- `GPTDatasetV1`, `GPTDatasetExamples`: build tokenized chunks for language modeling
- `create_dataloader_from_examples(...)`: construct PyTorch dataloaders
- `train_model_simple(...)`: trains for `num_epochs` with periodic evaluation and text sampling

Artifacts:
- Final checkpoint saved as `last_model.pt` (notebook shows where this is written)
- Additional weights from past runs: `gpt2_small_trained.pth`, `fine_tuned.pth`

Hardware notes:
- GPU strongly recommended; the notebook will automatically use CUDA if available

## Inference
The notebook provides utilities for text generation:
- `text_to_token_ids(...)`, `token_ids_to_text(...)`
- `generate_text_simple(model, idx, max_new_tokens, context_size)`

Minimal example (inside the notebook after training):
```python
import torch, tiktoken

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

# Load the trained model (paths depend on what you saved)
model = torch.load("last_model.pt", map_location=device)
model.eval()

start = "Every effort moves you"
encoded = torch.tensor(tokenizer.encode(start)).unsqueeze(0).to(device)

with torch.no_grad():
    out = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=1024)

print(tokenizer.decode(out.squeeze(0).tolist()))
```

Note: If you trained a custom architecture defined inside the notebook, load the state dict into the same model class you instantiated there rather than using `torch.load` on the whole object.

## Tips
- Start with short context length and small batch size when testing
- Inspect token counts and apply length filters to avoid over‑long sequences
- Regularly print generated samples to sanity‑check training progress

## Caveats and attribution
- This repository is for educational purposes. Do not use generated medical content for clinical decisions
- Datasets referenced: MedQuAD (via `medquad.csv`) and `nvidia/HelpSteer` (Hugging Face). Respect their licenses/terms
- `gpt2/355M` contains original GPT‑2 reference assets; they are not used by default in the notebook



