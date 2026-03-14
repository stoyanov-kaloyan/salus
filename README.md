# Salus

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

### Manual CLI usage

```
# Analyze a specific image
python .\deterministic_analysis.py .\test_images\real1.jpg
python .\deterministic_analysis.py .\test_images\deepfake1.png

# Runtime profile controls (default profile is now "fast")
$env:DETERMINISTIC_PROFILE = "fast"      # fast | balanced | full
$env:DETERMINISTIC_MAX_SIDE = "512"      # resize cap before deterministic analysis

# Use a trained calibration file
$env:DETERMINISTIC_CALIBRATION_PATH = ".\deterministic_calibration.json"
python .\deterministic_analysis.py .\test_images\real1.jpg
```

### Deterministic calibration training

```
# manifest.csv columns: path,label  (0=real, 1=deepfake)
python .\deterministic_training.py `
  --manifest .\manifest.csv `
  --output .\deterministic_calibration.json `
  --profile fast `
  --epochs 1200 `
  --validation-fraction 0.20 `
  --max-workers 4
```

The deterministic runtime uses the `fast` profile by default to keep request latency down. If you train a calibration file, use the same profile and image-size cap in both training and serving so the learned weights match the live feature vectors.
