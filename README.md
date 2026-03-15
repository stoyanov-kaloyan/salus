# Salus

Deepfake image protection.

## How to set up Iphone proxy.

### Run this and go to http://mitm.it

```
 mitmdump -s proxy.py --listen-host 0.0.0.0 --listen-port 8080
```

## Run the FastAPI server

```
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Phone setup

1. Wi-Fi settings > connect to wifi > Configure Proxy > Manual

- Server: ip of the server
- Port: port of the server

2. Go to `http://mitm.it` and download the sertificate for iOS
3. General > About > Certificate Trust Settings > Enable full trust for root certificates

## Available filters

- NSFW filter
- Deepfake filter
- Domain filter
- Harmful text filter

## Unit Tests for python modules

```
# All tests
python -m unittest discover -s tests

# Hybrid pipeline (deterministic multiview + neural fusion logic)
python -m unittest tests/test_hybrid_risk_pipeline.py

# GPU integration test (requires transformers + a real GPU)
python -m unittest tests/test_deepfake_recognizer_gpu_integration.py
```

## Compile Presentation (Quarto)

requires `quarto` CLI installed

Presentation source files are in `presentation/`:

- `presentation/salus_presentation.qmd`
- `presentation/generate_charts.py`

### 1) Generate chart assets (matplotlib)

From repo root:

```powershell
# optional: activate local venv
& .\.venv\Scripts\Activate.ps1

# build charts used in slides
python .\presentation\generate_charts.py
```

This writes image assets and summary metrics to `presentation/assets/`.

### 2) Render slides with Quarto

From repo root:

```powershell
# RevealJS HTML slides
quarto render presentation/salus_presentation.qmd --to revealjs

# PowerPoint export
quarto render presentation/salus_presentation.qmd --to pptx
```

Generated files:

- `presentation/salus_presentation.html`
- `presentation/salus_presentation.pptx`

### 3) Optional: render both in one command

```powershell
quarto render presentation/salus_presentation.qmd --to revealjs; quarto render presentation/salus_presentation.qmd --to pptx
```
