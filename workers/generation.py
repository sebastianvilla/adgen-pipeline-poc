import os
import base64
from io import BytesIO
from datetime import datetime
from PIL import Image
from openai import OpenAI

# Create a single client (re-use across calls)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def _to_png_buffer(im: Image.Image, name: str) -> BytesIO:

    buf = BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    buf.name = name
    return buf
 
def generate(prompt: str,
             product: str,
             region: str,
             output_dir: str,
             output_name: str,
             size: str,
             input_image_path: str,  
             mask_image_path: str | None = None) -> str:

    src_img = Image.open(input_image_path).convert("RGBA")
    img_buf = _to_png_buffer(src_img, "image.png")

    mask_buf = None
    if mask_image_path and os.path.exists(mask_image_path):
        mask_img = Image.open(mask_image_path).convert("RGBA")
        mask_buf = _to_png_buffer(mask_img, "mask.png")

    kwargs = dict(
        model="gpt-image-1",
        image=img_buf,
        prompt=prompt,
        background="opaque",
        size=size,
        input_fidelity="high",
        output_format="png",
    )

    if mask_buf:
        kwargs["mask"] = mask_buf

    # Call OpenAI Images Edit
    try:
        resp = client.images.edit(**kwargs)
    except Exception as e:
        raise RuntimeError(f"OpenAI Images API error: {e}")

    if not resp or not resp.data or not resp.data[0].b64_json:
        raise RuntimeError("OpenAI Images API returned an empty response.")

    # Decode and save
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, output_name)
    png_bytes = base64.b64decode(resp.data[0].b64_json)
    with open(image_path, "wb") as f:
        f.write(png_bytes)

    return image_path