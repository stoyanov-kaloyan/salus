import mitmproxy.http
from mitmproxy import ctx
from PIL import Image
import io
import cv2
import numpy as np
import os

from recognizers import create_deepfake_recognizer

class ImageBlurAddon:
    def __init__(self):
        device = int(os.getenv("DEEPFAKE_PIPELINE_DEVICE", "0"))
        threshold = float(os.getenv("DEEPFAKE_THRESHOLD", "0.5"))
        backend = os.getenv("DEEPFAKE_MODEL_BACKEND", "auto")
        checkpoint_path = os.getenv("DEEPFAKE_PT_CHECKPOINT_PATH")
        self.recognizer = create_deepfake_recognizer(
            backend=backend,
            checkpoint_path=checkpoint_path,
            device=device,
            threshold=threshold,
        )

    def request(self, flow: mitmproxy.http.HTTPFlow):
        # Log all requests to verify traffic is flowing
        ctx.log.info(f"Request: {flow.request.url}")
    
    def response(self, flow: mitmproxy.http.HTTPFlow):
        content_type = flow.response.headers.get("content-type", "")
        
        ctx.log.info(f"Response from {flow.request.url} - Content-Type: {content_type}")
        
        # Only process image responses
        if not content_type.startswith("image/"):
            return
            
        ctx.log.info(f"Processing image from {flow.request.url}")
        
        try:
            # Get image bytes
            image_data = flow.response.content
            
            # Load with PIL
            img = Image.open(io.BytesIO(image_data))
            image_for_recognizer = img.convert("RGB")
            
            if not self.should_blur(image_for_recognizer):
                return

            ctx.log.info(f"Blurring image from {flow.request.url}")
            blurred = cv2.GaussianBlur(np.array(image_for_recognizer), (51, 51), 0)
            blurred_img = Image.fromarray(blurred)
            
            # Write back into the response
            output = io.BytesIO()
            blurred_img.save(output, format=img.format or "JPEG")
            flow.response.content = output.getvalue()
            
        except Exception as e:
            ctx.log.warn(f"Image processing failed: {e}")
    
    def should_blur(self, image: Image.Image) -> bool:
        try:
            decision = self.recognizer.evaluate(image)
            ctx.log.info(
                f"Recognizer decision: label={decision.label}, score={decision.score:.4f}, "
                f"threshold={self.recognizer.threshold:.2f}, should_blur={decision.should_change}"
            )
            return decision.should_change
        except Exception as e:
            ctx.log.warn(f"Recognizer failed: {e}")
            return False

addons = [ImageBlurAddon()]