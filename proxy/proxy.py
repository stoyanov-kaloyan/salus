from __future__ import annotations

import base64
import io
import os
import uuid
from pathlib import Path

import httpx
from mitmproxy import ctx, http
from PIL import Image

SETUP_HOST = os.getenv("PROXY_SETUP_HOST", "salus.proxy")
MIN_IMAGE_BYTES = int(os.getenv("MIN_IMAGE_BYTES", str(10 * 1024)))
DETECTION_API_URL = os.getenv("DETECTION_API_URL", "http://localhost:8000")
DETECTION_RECOGNIZERS = os.getenv("DETECTION_RECOGNIZERS", "deepfake,nsfw")

_cover_bytes: bytes = b""


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
