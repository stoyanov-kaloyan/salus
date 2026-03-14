# Salus Proxy

An HTTP/HTTPS intercepting proxy that blurs images in transit. Devices route their traffic through it; the proxy decrypts HTTPS using a locally-trusted CA, processes image responses, then re-encrypts before forwarding to the client.

## How it works

```
Device → [HTTP proxy] → salus proxy → Internet
                           ↓
                    MITM decrypt HTTPS
                    blur image responses
                    re-encrypt to client
```

**HTTPS interception** is done via CONNECT tunneling. When a client opens a CONNECT tunnel, the proxy performs a man-in-the-middle: it presents a dynamically signed leaf certificate to the client (signed by the Salus CA), and opens its own TLS connection upstream. This is the standard approach for SSL inspection.

**Image blurring** happens on the response path. Any response with a `Content-Type: image/*` header gets its body read, decoded, blurred, and re-encoded before being forwarded. The blur is a box blur run 3 times, which approximates a Gaussian. WebP images are re-encoded as JPEG since the Go standard library has no WebP encoder.

**CA management** — on first run, the proxy generates a self-signed ECDSA P-256 CA and writes it to `certs/ca.crt` and `certs/ca.key`. The CA is valid for 10 years. If the files already exist they're loaded as-is.

## Device setup

Point the device's HTTP proxy at the proxy's address, then install the CA certificate so it trusts the proxy's TLS certificates. Navigate to `http://salus.proxy` (while the proxy is active) to reach the setup page, which detects your platform and offers the right certificate format:

| Path | Format | Platform |
|------|--------|----------|
| `/ios` | `.mobileconfig` | iOS / iPadOS |
| `/android` | `.crt` (DER) | Android |
| `/pem` | `.pem` | macOS, Windows, Linux |

The setup page also shows step-by-step installation instructions per platform.

## Configuration

All settings are environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_ADDR` | `:8080` | Listen address |
| `PROXY_CA_CERT` | `certs/ca.crt` | CA certificate path |
| `PROXY_CA_KEY` | `certs/ca.key` | CA private key path |
| `PROXY_BLUR_RADIUS` | `25` | Box blur radius in pixels |
| `PROXY_DETECTION_API_URL` | `` | Detection API endpoint (see below) |
| `PROXY_SETUP_HOST` | `salus.proxy` | Hostname for the setup page |
| `PROXY_VERBOSE` | `false` | Enable debug logging |

## Detection API

When `PROXY_DETECTION_API_URL` is set, images will eventually be sent to that endpoint to decide whether to blur them. Currently this is not implemented — all images are blurred regardless.

## Running

```sh
# Local
go run .

# Docker
docker build -t salus-proxy .
docker run -p 8080:8080 -v $(pwd)/certs:/certs salus-proxy
```

The CA certificate is served at `http://<proxy-addr>/ca.crt` as well as through the setup page.
