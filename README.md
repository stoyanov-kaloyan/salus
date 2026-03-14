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
