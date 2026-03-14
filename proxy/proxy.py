"""
Salus proxy — mitmproxy addon.

Handles:
  1. Setup page serving (cert downloads at http://salus.proxy/)
  2. AI-based image detection + replacement
"""
from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path

import httpx
from mitmproxy import ctx, http
from PIL import Image

# ── Environment ──────────────────────────────────────────────────────────────

SETUP_HOST = os.getenv("PROXY_SETUP_HOST", "salus.proxy")
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", str(10 * 1024)))  # skip images < 10 KB
DETECTION_API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8000")
DETECTION_RECOGNIZERS = os.getenv("DETECTION_RECOGNIZERS", "deepfake,nsfw")

# ── Module-level singletons (populated in load()) ────────────────────────────

_cover_bytes: bytes = b""

# ── Setup page ───────────────────────────────────────────────────────────────


def _read_ca_pem() -> bytes:
    confdir = ctx.options.confdir
    pem_path = Path(confdir).expanduser() / "mitmproxy-ca-cert.pem"
    if pem_path.exists():
        return pem_path.read_bytes()
    return b""


def _pem_to_der(pem: bytes) -> bytes:
    lines = pem.decode("ascii", errors="replace").splitlines()
    b64 = "".join(line for line in lines if not line.startswith("-----"))
    return base64.b64decode(b64)


def _build_mobileconfig(pem: bytes) -> bytes:
    der = _pem_to_der(pem)
    b64 = base64.b64encode(der).decode("ascii")
    indented = "\n".join(
        "        " + b64[i : i + 64] for i in range(0, len(b64), 64)
    )
    profile_uuid = str(uuid.uuid4())
    cert_uuid = str(uuid.uuid4())
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>PayloadVersion</key><integer>1</integer>
  <key>PayloadUUID</key><string>{profile_uuid}</string>
  <key>PayloadType</key><string>Configuration</string>
  <key>PayloadIdentifier</key><string>io.salus.proxy.ca</string>
  <key>PayloadDisplayName</key><string>Salus Proxy CA</string>
  <key>PayloadDescription</key><string>Allows Salus proxy to inspect HTTPS traffic</string>
  <key>PayloadOrganization</key><string>Salus</string>
  <key>PayloadContent</key>
  <array>
    <dict>
      <key>PayloadType</key><string>com.apple.security.root</string>
      <key>PayloadVersion</key><integer>1</integer>
      <key>PayloadIdentifier</key><string>io.salus.proxy.ca.root</string>
      <key>PayloadUUID</key><string>{cert_uuid}</string>
      <key>PayloadDisplayName</key><string>Salus Proxy CA</string>
      <key>PayloadContent</key>
      <data>
{indented}
      </data>
    </dict>
  </array>
</dict>
</plist>""".encode(
        "utf-8"
    )


def _detect_platform(ua: str) -> str:
    ua_lower = ua.lower()
    if "iphone" in ua_lower or "ipad" in ua_lower:
        return "ios"
    if "android" in ua_lower:
        return "android"
    return "desktop"


def _build_setup_html(platform: str) -> str:
    all_buttons = [
        ("/ios", "iOS / iPadOS (.mobileconfig)"),
        ("/android", "Android (.crt)"),
        ("/pem", "macOS / Windows / Linux (.pem)"),
    ]
    primary_map = {"ios": 0, "android": 1, "desktop": 2}
    primary_idx = primary_map.get(platform, 2)

    btn_parts = []
    for i, (href, label) in enumerate(all_buttons):
        cls = "btn btn-primary" if i == primary_idx else "btn"
        btn_parts.append(f'<a href="{href}" class="{cls}">{label}</a>')
    primary_btn = btn_parts.pop(primary_idx)
    btn_parts.insert(0, primary_btn)
    btn_html = "\n".join(btn_parts)

    if platform == "ios":
        instructions = """<ol>
  <li>Tap <strong>iOS (.mobileconfig)</strong> above and confirm the download.</li>
  <li>Open <strong>Settings → General → VPN &amp; Device Management</strong>.</li>
  <li>Tap <em>Salus Proxy CA</em> → <strong>Install</strong>.</li>
  <li>Go to <strong>Settings → General → About → Certificate Trust Settings</strong>.</li>
  <li>Enable full trust for <em>Salus Proxy CA</em>.</li>
</ol>"""
    elif platform == "android":
        instructions = """<ol>
  <li>Tap <strong>Android (.crt)</strong> above to download the certificate.</li>
  <li>Open <strong>Settings → Security → Install a certificate → CA certificate</strong>.</li>
  <li>Select the downloaded file and confirm installation.</li>
</ol>"""
    else:
        instructions = """<ol>
  <li>Download the <strong>.pem</strong> certificate above.</li>
  <li><strong>macOS</strong>: Double-click the file → Keychain Access → set to <em>Always Trust</em>.</li>
  <li><strong>Windows</strong>: Double-click → Install Certificate → Trusted Root Certification Authorities.</li>
  <li><strong>Linux / Chrome</strong>: <code>sudo cp ca.pem /usr/local/share/ca-certificates/salus-proxy.crt &amp;&amp; sudo update-ca-certificates</code></li>
</ol>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Salus Proxy Setup</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 600px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1   {{ font-size: 1.6rem; margin-bottom: 0.3em; }}
  p    {{ color: #555; }}
  .btn {{ display: block; margin: 10px 0; padding: 14px 20px; border-radius: 8px;
         text-decoration: none; font-size: 1rem; text-align: center;
         background: #e8e8e8; color: #333; }}
  .btn-primary {{ background: #0071e3; color: #fff; font-weight: 600; }}
  .instructions {{ margin-top: 30px; background: #f5f5f5; border-radius: 8px;
                  padding: 16px 20px; }}
  .instructions h2 {{ margin-top: 0; font-size: 1rem; }}
  code {{ background: #e0e0e0; padding: 2px 5px; border-radius: 4px; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Salus Proxy Setup</h1>
<p>Install the CA certificate on your device to allow the proxy to inspect HTTPS traffic.</p>
{btn_html}
<div class="instructions">
  <h2>Installation steps</h2>
{instructions}
</div>
</body>
</html>"""


def _serve_setup(flow: http.HTTPFlow) -> None:
    path = flow.request.path.split("?")[0]
    pem = _read_ca_pem()

    if path == "/ios":
        body = (
            _build_mobileconfig(pem)
            if pem
            else b"CA cert not yet generated \xe2\x80\x94 try again in a moment."
        )
        flow.response = http.Response.make(
            200,
            body,
            {
                "Content-Type": "application/x-apple-aspen-config",
                "Content-Disposition": 'attachment; filename="salus-proxy.mobileconfig"',
            },
        )
    elif path == "/android":
        body = _pem_to_der(pem) if pem else b"CA cert not yet generated."
        flow.response = http.Response.make(
            200,
            body,
            {
                "Content-Type": "application/x-x509-ca-cert",
                "Content-Disposition": 'attachment; filename="salus-proxy-ca.crt"',
            },
        )
    elif path == "/pem":
        body = pem if pem else b"CA cert not yet generated."
        flow.response = http.Response.make(
            200,
            body,
            {
                "Content-Type": "application/x-pem-file",
                "Content-Disposition": 'attachment; filename="salus-proxy-ca.pem"',
            },
        )
    else:
        ua = flow.request.headers.get("user-agent", "")
        platform = _detect_platform(ua)
        body = _build_setup_html(platform).encode("utf-8")
        flow.response = http.Response.make(
            200,
            body,
            {"Content-Type": "text/html; charset=utf-8"},
        )


# ── mitmproxy addon ───────────────────────────────────────────────────────────


class SalusAddon:

    def load(self, loader) -> None:
        global _cover_bytes

        cover_path = Path(__file__).parent / "static" / "cover.jpg"
        if cover_path.exists():
            _cover_bytes = cover_path.read_bytes()
            ctx.log.info(f"[salus] Cover image loaded ({len(_cover_bytes)} bytes)")
        else:
            ctx.log.warn(f"[salus] Cover image not found at {cover_path}")

        ctx.log.info(f"[salus] Detection API URL: {DETECTION_API_URL}")

    def request(self, flow: http.HTTPFlow) -> None:
        if flow.request.pretty_host == SETUP_HOST:
            _serve_setup(flow)

    async def response(self, flow: http.HTTPFlow) -> None:
        if flow.response is None:
            return

        content_type = flow.response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return

        if not _cover_bytes:
            return

        raw = flow.response.get_content()
        if not raw or len(raw) < MIN_IMAGE_BYTES:
            return

        try:
            Image.open(io.BytesIO(raw))
        except Exception:
            return

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{DETECTION_API_URL}/detect/image",
                    files={"file": ("image", raw, content_type)},
                    data={"recognizers": DETECTION_RECOGNIZERS},
                    timeout=10.0,
                )
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            ctx.log.warn(f"[salus] Detection API error for {flow.request.pretty_url}: {exc}")
            return

        results = payload.get("results", {})
        if any(v.get("is_target") for v in results.values()):
            triggered = [k for k, v in results.items() if v.get("is_target")]
            scores = {k: v.get("score", 0) for k, v in results.items() if v.get("is_target")}
            ctx.log.info(
                f"[salus] Replacing image ({triggered} scores={scores}): "
                f"{flow.request.pretty_url}"
            )
            flow.response.headers["content-type"] = "image/jpeg"
            if "content-encoding" in flow.response.headers:
                del flow.response.headers["content-encoding"]
            flow.response.set_content(_cover_bytes)


addons = [SalusAddon()]
