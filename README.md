# The Visual ROI

Code for the master's thesis **"The Visual ROI: Quantifying the Impact of AI-Derived Visual Sentiment on Social Media Engagement and Revenue Performance"**
(Honest Albert Temu, MSc Data Science for Economics, Università degli Studi di Milano, 2025-2026;
supervisor Prof. Dario Malchiodi, co-supervisor Prof. Domenico Massaro).

The study scores 79 days of Instagram content of the streetwear brand Cold Culture (1 October to 18 December 2025) with the
zero-shot vision-language model CLIP ViT-B/32 and relates the daily visual score to daily revenue with OLS marketing mix models.

## Contents

| Path | What it is |
|---|---|
| `Thesis.ipynb` | The analysis notebook: every number, figure and table in the thesis |
| `requirements.txt` | Pinned packages for the analysis (Stages 2-5) |
| `requirements-clip.txt` | Pinned packages for the optional CLIP scoring stage (Stage 1) |
| `figures/` | Thesis figures as produced by the notebook |
| `results/tables/` | Thesis tables as CSV |
| `results/runtime.json` | Hardware, software versions and stage timings (written by Stage 5) |

## Data

The brand's media archive and sales records, and the analysis dataset `master_dataset.csv` derived from them, are **not** included.
They are available from the author on reasonable request, subject to the brand's permission. With the file in the repository root,
the notebook checks its SHA-256 fingerprint against the dataset used in the thesis.

## Running the notebook

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook Thesis.ipynb        # or open it in VS Code / Google Colab and choose "Run all"
```

With the default switches (`RUN_CLIP_SCORING = False`, `REBUILD_MASTER = False`) the notebook reads `master_dataset.csv`,
estimates Models 1-4 with OLS and HC3 standard errors, writes the figures and tables, and ends with a check that should print
`57 of 57 numbers match the thesis.`

To re-score the media with CLIP, place the archive at `data/cculture.zip` (media files named `<anything>_YYYY-MM-DD.<ext>`),
install `requirements-clip.txt`, and set `RUN_CLIP_SCORING = True` (and `REBUILD_MASTER = True` to rebuild the dataset).

## Model

CLIP ViT-B/32 (Radford et al., 2021), loaded through OpenAI's reference implementation
(<https://github.com/openai/CLIP>, commit `d05afc4`). The same checkpoint is available on Hugging Face as
[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32).
Each image or video frame (every 60th frame) is compared with two prompts, and the visual score is the softmax probability of the
product-focused prompt:

1. "A clear fashion photography shot where the streetwear clothing and outfit are the obvious central focus."
2. "An atmospheric lifestyle, scenery, or mood photo where the clothing is obscured, hidden, or not the main subject."
