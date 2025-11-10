from PIL import Image, ImageDraw, ImageFont, ImageStat
import os

# ---------------- Brand colors & priorities ----------------

BRAND_PALETTES = {
    # 'text_priority' is a list of named colors in the order we want to try first.
    # We'll only fall back to others if contrast is insufficient.
    "cocacola_zero_cherry": {
        "red":       (230, 30, 43),
        "white":     (255, 255, 255),
        "black":     (0, 0, 0),
        "text_priority": ["red", "white", "black"]
    },
    "sprite_lemon_lime": {
        "green":     (0, 140, 70),
        "white":     (255, 255, 255),
        "black":     (0, 0, 0),
        "text_priority": ["green", "white", "black"]
    }
}

# ---------------- Tiny helpers ----------------

def _font_path_for_region(region: str, font_dir="assets/fonts"):
    os.makedirs(font_dir, exist_ok=True)
    if (region or "").lower() == "ja-jp":
        # cand = os.path.join(font_dir, "NotoSansJP-Bold.ttf")
        cand = os.path.join(font_dir, "NotoSansJP-ExtraBold.ttf")
    else:
        # cand = os.path.join(font_dir, "NotoSans-Bold.ttf")
        cand = os.path.join(font_dir, "NotoSans-ExtraBold.ttf")
    return cand if os.path.exists(cand) else None

def _load_font(region: str, size: int, font_dir="assets/fonts"):
    p = _font_path_for_region(region, font_dir)
    try:
        if p:
            return ImageFont.truetype(p, size)
    except Exception:
        pass
    return ImageFont.load_default()

def _measure(draw, s, font):
    bbox = draw.textbbox((0,0), s, font=font)
    return bbox[2]-bbox[0], bbox[3]-bbox[1]

def _wrap_to_width(draw, text, font, max_width, line_spacing=6):
    words = text.split()
    if not words:
        return "", (0, 0)
    lines, cur = [], []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        tw, th = draw.textbbox((0,0), test, font=font)[2:]
        if not cur or tw <= max_width:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    wrapped = "\n".join(lines)
    bbox = draw.multiline_textbbox((0,0), wrapped, font=font, spacing=line_spacing, align="center")
    return wrapped, (bbox[2]-bbox[0], bbox[3]-bbox[1])

def _tokenize_for_wrap(text: str, region: str):
    is_ja = (region or "").lower().startswith("ja")
    if is_ja and (" " not in text):
        return list(text), True
    return text.split(), False

def _rebalance_widow(lines, draw, font, max_width, min_last_words=3, min_last_frac=0.60):
    changed = True
    while changed and len(lines) >= 2:
        changed = False
        last = lines[-1].strip()
        prev = lines[-2].strip()
        if not last or not prev:
            break
        last_words = len(last.split())
        w_last, _ = _measure(draw, last, font)
        w_prev, _ = _measure(draw, prev, font)
        widowish = (last_words <= min_last_words) or (w_last < min_last_frac * w_prev)
        if widowish:
            merged = f"{prev} {last}"
            w_merged, _ = _measure(draw, merged, font)
            if w_merged <= max_width:
                lines[-2] = merged
                lines.pop()
                changed = True
    return lines

def _wrap_to_width_balanced(draw, text, font, max_width, align="center", line_spacing=6, region=""):
    tokens, join_without_space = _tokenize_for_wrap(text, region)
    if not tokens:
        return "", (0, 0)
    def join(seq): return "".join(seq) if join_without_space else " ".join(seq)
    lines, cur = [], []
    for tk in tokens:
        test = join(cur + [tk]).strip()
        tw, _ = _measure(draw, test, font)
        if not cur or tw <= max_width:
            cur.append(tk)
        else:
            lines.append(join(cur))
            cur = [tk]
    if cur:
        lines.append(join(cur))
    if not join_without_space:
        lines = _rebalance_widow(lines, draw, font, max_width)
    wrapped = "\n".join(lines)
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, align=align, spacing=line_spacing)
    return wrapped, (bbox[2] - bbox[0], bbox[3] - bbox[1])

def _fit_text(draw, text, box_w, box_h, region, min_size=14, max_size=200, line_spacing=6):
    lo, hi = min_size, max_size
    best = (ImageFont.load_default(), text, (0,0))
    if (region or "").lower().startswith("ja"):
        max_size = int(max_size * 1.15)
    while lo <= hi:
        mid = (lo + hi)//2
        font = _load_font(region, mid)
        wrapped, (tw, th) = _wrap_to_width_balanced(draw, text, font, box_w, align="center",
                                                    line_spacing=line_spacing, region=region)
        if tw <= box_w and th <= box_h:
            best = (font, wrapped, (tw, th))
            lo = mid + 1
        else:
            hi = mid - 1
    return best

def _cover_resize_center_crop(img, target_w, target_h):
    sw, sh = img.size
    if sw/sh > target_w/target_h:
        scale = target_h / sh
        nw, nh = int(sw*scale), target_h
    else:
        scale = target_w / sw
        nw, nh = target_w, int(sh*scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left = (nw - target_w)//2
    top  = (nh - target_h)//2
    return img.crop((left, top, left+target_w, top+target_h))

# ---------------- Contrast utilities ----------------

def _rel_lum(rgb):
    c = [x/255.0 for x in rgb]
    c = [v/12.92 if v <= 0.03928 else ((v+0.055)/1.055)**2.4 for v in c]
    return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]

def _contrast_ratio(rgb1, rgb2):
    L1, L2 = _rel_lum(rgb1), _rel_lum(rgb2)
    L1, L2 = max(L1, L2), min(L1, L2)
    return (L1 + 0.05) / (L2 + 0.05)

def _avg_rgb(img_rgba):
    r, g, b = ImageStat.Stat(img_rgba.convert("RGB")).mean
    return (int(r), int(g), int(b))

def _choose_brand_text_color_CONTRAST(brand: str, bg_crop_rgba, min_contrast=3.5):
    
    palette = BRAND_PALETTES.get(brand)
    bg_rgb = _avg_rgb(bg_crop_rgba)

    # Build candidate list in priority order
    candidates = []
    for name in palette.get("text_priority", []):
        if name in palette:
            candidates.append((name, palette[name]))
    # Ensure we always consider white/black as last resorts
    for name in ("white", "black"):
        if name in palette and name not in [n for n,_ in candidates]:
            candidates.append((name, palette[name]))

    best = None
    best_cr = -1.0
    for _, rgb in candidates:
        cr = _contrast_ratio(rgb, bg_rgb)
        if cr >= min_contrast:
            return rgb
        if cr > best_cr:
            best, best_cr = rgb, cr
    return best  # highest-contrast fallback

def _avg_luminance(img_rgba):
    """Perceived luminance 0–255 using Rec. 709 coefficients."""
    r, g, b = ImageStat.Stat(img_rgba.convert("RGB")).mean
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _choose_brand_text_color(product, bg_crop_rgba):
    """Simpler heuristic: white on dark, brand primary on light."""

    # pick the palette
    palette = BRAND_PALETTES.get(product)
    primary = palette[palette["text_priority"][0]]  # first listed
    white = (255, 255, 255)
    black = (0, 0, 0)

    lum = _avg_luminance(bg_crop_rgba)

    # 0 = black, 255 = white
    if lum < 158:
        return white  # dark background → white text
    elif lum < 220:
        return primary  # medium-light → brand color
    else:
        # return black  # very bright → black text
        return primary  # very bright → black text

# ---------------- Main single-pass composer ----------------

def compose(image_path, message, ratio_str, logo_path, product, region, output_dir, output_name):
    """
    Creative composer, handles fitment and adaptive text color.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")
    if logo_path and not os.path.exists(logo_path):
        raise ValueError(f"Logo path does not exist: {logo_path}")
    os.makedirs(output_dir, exist_ok=True)

    hero = Image.open(image_path).convert("RGBA")
    logo = Image.open(logo_path).convert("RGBA") if logo_path else None

    # Target size (long side = 1024)
    rw, rh = map(int, ratio_str.split(":"))
    if rw >= rh:
        W = 1024; H = max(1, int(1024 * rh / rw))
    else:
        H = 1024; W = max(1, int(1024 * rw / rh))

    canvas = _cover_resize_center_crop(hero, W, H).convert("RGBA")
    draw = ImageDraw.Draw(canvas)

    # Logo (top-left)
    margin = int(min(W, H) * 0.05)
    if logo:
        lw = int(W * 0.16)
        ratio = max(1e-6, logo.width / logo.height)
        logo_r = logo.resize((lw, max(1, int(lw/ratio))), Image.LANCZOS)
        canvas.paste(logo_r, (margin, margin), logo_r)

    # Text box presets
    if ratio_str == "1:1":
        box_w = int(W * 0.90); box_h = int(H * 0.28)
        x = (W - box_w)//2; y = H - box_h - int(H*0.06)
        align = "center"
    elif ratio_str == "16:9":
        box_w = int(W * 0.40); box_h = int(H * 0.52)
        x = int(W * 0.08); y = H - box_h - int(H*0.14)
        align = "left"
    elif ratio_str == "9:16":
        box_w = int(W * 0.84); box_h = int(H * 0.30)
        x = (W - box_w)//2; y = H - box_h - int(H*0.10)
        align = "center"
    else:
        box_w = int(W * 0.90); box_h = int(H * 0.28)
        x = (W - box_w)//2; y = H - box_h - int(H*0.06)
        align = "center"

    if message:
        pad = int(min(box_w, box_h) * 0.10)
        inner_w = max(1, box_w - 2*pad)
        inner_h = max(1, box_h - 2*pad)

        line_spacing = 4

        font, wrapped, (tw, th) = _fit_text(draw, message, inner_w, inner_h,
                                            region=region, min_size=14, max_size=90, line_spacing=line_spacing)

        # Sample ONLY where the text will be drawn (+ small padding)
        sample_pad = max(6, pad//2)
        sx0 = max(0, x + pad - sample_pad)   
        sy0 = max(0, y + pad - sample_pad)
        sx1 = min(W, x + box_w - pad + sample_pad)
        sy1 = min(H, y + box_h - pad + sample_pad)
        bg_sample = canvas.crop((sx0, sy0, sx1, sy1)) 

        # fill_col = _choose_brand_text_color(product, bg_sample, min_contrast=3.0)
        fill_col = _choose_brand_text_color(product, bg_sample) 

        # Auto stroke for legibility (opposite polarity)
        stroke_col = (200, 200, 200) if sum(fill_col)/3 > 128 else (50, 50, 50)

        tx = x + (box_w - tw)//2 if align == "center" else x + pad
        ty = y + (box_h - th)//2

        draw.multiline_text(
            (tx, ty), wrapped, font=font, fill=fill_col, align=align, spacing=line_spacing,
            stroke_width=0, stroke_fill=stroke_col,
        )

    out_path = os.path.join(output_dir, output_name)
    canvas.save(out_path, "PNG", optimize=True)
    return out_path
