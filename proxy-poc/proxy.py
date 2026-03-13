import mitmproxy.http
from mitmproxy import ctx
from PIL import Image
import io
import cv2
import numpy as np

class ImageBlurAddon:
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
            img_array = np.array(img)
            
            # --- Your detection logic here ---
            if self.should_blur(img_array):
                ctx.log.info(f"Blurring image from {flow.request.url}")
                blurred = cv2.GaussianBlur(img_array, (51, 51), 0)
                img = Image.fromarray(blurred)
            
            # Write back into the response
            output = io.BytesIO()
            img.save(output, format=img.format or "JPEG")
            flow.response.content = output.getvalue()
            
        except Exception as e:
            ctx.log.warn(f"Image processing failed: {e}")
    
    def should_blur(self, img_array):
        # Placeholder — plug in any model here
        return True

addons = [ImageBlurAddon()]