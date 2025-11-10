import os
import json
import logging
from uuid import uuid4
import base64
from datetime import datetime
from PIL import Image, ImageOps
from workers.legal_check import check as legal_check
from workers.generation import generate as generate_image
from workers.prompt_builder import generate as generate_prompt
from workers.translation import translate as translate_message
from workers.composition import compose as compose_creatives
from workers.brand_compliance import check as brand_compliance
import shutil
import logging
from logging import FileHandler, Formatter

ASSETS_DIR = "assets"

def _save_provided_asset(output_dir: str, product: str, asset_data: str) -> str | None:

    if not asset_data:
        return None

    dest_dir = os.path.join(output_dir, "assets")
    _ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, f"{product}.png")

    # data URL
    if asset_data.startswith("data:image"):
        try:
            header, data = asset_data.split(",", 1)
            image_data = base64.b64decode(data)
            with open(dest_path, "wb") as f:
                f.write(image_data)
            return dest_path
        except Exception:
            return None

    # filesystem path
    if os.path.exists(asset_data):
        try:
            shutil.copyfile(asset_data, dest_path)
            return dest_path
        except Exception:
            return None

    return None

def _aspect_ratio_to_size(aspect_ratio: str) -> str:
    ar = aspect_ratio.strip()
    if ar == "1:1": return "1024x1024"
    if ar in ("9:16", "9/16"): return "1024x1536"
    if ar in ("16:9", "16/9"): return "1536x1024"
    return "1024x1024"

def _resolve_asset_from_base64_or_path(output_dir: str, product: str, region: str, asset_data: str) -> str | None:
    if not asset_data: return None
    if asset_data.startswith("data:image"):
        header, data = asset_data.split(",", 1)
        image_data = base64.b64decode(data)
        asset_path = os.path.join(output_dir, f"{product}_{region}_input_hero.png")
        with open(asset_path, "wb") as f: f.write(image_data)
        return asset_path
    if os.path.exists(asset_data): return asset_data
    return None

def _resolve_product_asset_paths(product: str, region: str) -> tuple[str | None, str | None]:
    input_image = os.path.join(ASSETS_DIR, f"{product}__{region}.png")
    mask_image  = os.path.join(ASSETS_DIR, f"{product}__mask.png")
    if not os.path.exists(input_image): input_image = None
    if not os.path.exists(mask_image): mask_image = None
    return input_image, mask_image

def _ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def _build_white_canvas_with_1x1(base_1x1_path: str, out_path: str, target_w: int, target_h: int, placement: str):
    base = Image.open(base_1x1_path).convert("RGBA")
    # canvas = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
    canvas = Image.new("RGBA", (target_w, target_h), "#ff7e00")
    if placement == "right": x, y = target_w - base.width, 0
    elif placement == "top": x, y = 0, 0
    else: raise ValueError("placement must be 'right' or 'top'")
    canvas.paste(base, (x, y), base)
    canvas.save(out_path, "PNG")

def _crop_center_keep_size(img_path: str, out_path: str, final_w: int | None, final_h: int | None):
    img = Image.open(img_path).convert("RGBA")
    W, H = img.size
    if final_w is None and final_h is None: raise ValueError("final_w or final_h must be provided")
    if final_w is None:
        crop_h = final_h; top = max(0, (H - crop_h) // 2); box = (0, top, W, top + crop_h)
    elif final_h is None:
        crop_w = final_w; left = max(0, (W - crop_w) // 2); box = (left, 0, left + crop_w, H)
    else:
        left = max(0, (W - final_w) // 2); top = max(0, (H - final_h) // 2); box = (left, top, left + final_w, top + final_h)
    Image.open(img_path).convert("RGBA")  # ensure open
    cropped = Image.open(img_path).convert("RGBA").crop(box)
    cropped.save(out_path, "PNG")

def _now_iso(): return datetime.utcnow().isoformat() + "Z"

def _atomic_save_json(path: str, data: dict):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=4)
    os.replace(tmp, path)

def _init_progress(brief: dict) -> dict:
    per = {}
    total_steps = 0

    # group by product
    outputs_by_product: dict[str, dict[str, list[str]]] = {}
    checks_by_product: dict[str, dict] = {}

    for p in brief["products"]:
        prod_id = p["id"]
        per[prod_id] = {}
        outputs_by_product[prod_id] = {}
        checks_by_product[prod_id] = {}
        for region in p["regions"]:
            outputs_by_product[prod_id][region] = []
            steps_for_region = ["translate", "legal", "gen_1x1"]
            if "1:1" in p["aspect_ratios"]:
                steps_for_region.append("compose_1x1")
            if "16:9" in p["aspect_ratios"]:
                steps_for_region.extend(["derive_16_9", "compose_16_9"])
            if "9:16" in p["aspect_ratios"]:
                steps_for_region.extend(["derive_9_16", "compose_9_16"])
            per[prod_id][region] = {
                "steps": {s: {"status": "pending"} for s in steps_for_region},
                "compliance": {},
                "outputs": []
            }
            total_steps += len(steps_for_region)

    return {
        "job_id": "",
        "campaign_id": brief.get("id", "unknown"),
        "status": "running",
        "outputs": outputs_by_product,  # { product_id: { region: [paths...] } }
        "checks": checks_by_product,   
        "progress": {
            "total_steps": total_steps,
            "completed_steps": 0,
            "percent": 0,
            "last_updated": _now_iso(),
            "per": per
        }
    }

def _tick(report: dict, output_dir: str, product: str, region: str, step_key: str, status: str = "done", extra: dict | None = None):
    step = report["progress"]["per"][product][region]["steps"].get(step_key)
    if step is None:
        report["progress"]["per"][product][region]["steps"][step_key] = {"status": status, "ts": _now_iso()}
    else:
        step["status"] = status; step["ts"] = _now_iso()
        if extra: step.update(extra)

    if status == "done":
        report["progress"]["completed_steps"] += 1

    total = max(1, report["progress"]["total_steps"])
    report["progress"]["percent"] = round((report["progress"]["completed_steps"] / total) * 100, 2)
    report["progress"]["last_updated"] = _now_iso()
    _atomic_save_json(os.path.join(output_dir, "report.json"), report)


def _add_output(report: dict, output_dir: str, product: str, path: str, region: str):
    report["outputs"].setdefault(product, {})
    report["outputs"][product].setdefault(region, [])
    report["outputs"][product][region].append(path) 
    report["progress"]["per"][product]  # ensure product exists
    report["progress"]["last_updated"] = _now_iso()
    _atomic_save_json(os.path.join(output_dir, "report.json"), report)

def _set_overall_status(report: dict, output_dir: str, status: str):
    report["status"] = status
    report["progress"]["last_updated"] = _now_iso()
    _atomic_save_json(os.path.join(output_dir, "report.json"), report)

def _add_compliance(report: dict, output_dir: str, product: str, region: str, variant_path: str, ok: bool):
    per = report["progress"]["per"][product][region]
    per["compliance"][os.path.basename(variant_path)] = {"ok": bool(ok), "ts": _now_iso()}
    # report["checks"].setdefault(product, {})
    # report["checks"][product][variant_path] = bool(ok) 
    report["checks"][product][region]["brand"] = bool(ok) 
    report["progress"]["last_updated"] = _now_iso()
    _atomic_save_json(os.path.join(output_dir, "report.json"), report)

# ----------------- main manager -----------------
class CampaignManager:
    def __init__(self, job_id):
        self.job_id = job_id
        self.output_dir = f"outputs/{job_id}"
        _ensure_dir(self.output_dir)
        _ensure_dir(os.path.join(self.output_dir, "temp"))

        self.logger = logging.getLogger(f"campaign.{self.job_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # don’t bubble to uvicorn/root

        log_path = os.path.join(self.output_dir, "log.txt")
        
        if not any(isinstance(h, FileHandler) and getattr(h, "baseFilename", "").endswith("log.txt")
                   for h in self.logger.handlers):
            fh = FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)

    def process(self, brief):
        self.logger.info(f"Starting job {self.job_id} with brief: {brief}")

        report = _init_progress(brief)
        report["job_id"] = self.job_id
        _atomic_save_json(os.path.join(self.output_dir, "report.json"), report)

        for product_obj in brief['products']:
            product = product_obj['id']
            regions = product_obj['regions']
            audience = product_obj['audience']
            en_message = product_obj['message']
            theme = product_obj['theme']
            aspect_ratios = product_obj['aspect_ratios']
            assets = product_obj['assets']

            if "cocacola" in product.lower(): brand = "cocacola"
            elif "sprite" in product.lower(): brand = "sprite"
            else: brand = product.lower()

            logo_path = f"assets/{brand}_logo.png"

            # keep references for variations when image is provided (no need to recreate for each region)
            base16_9 = None
            base9_16 = None

            provided_asset_path = None
            if isinstance(assets, dict):
                provided_asset_path = _save_provided_asset(self.output_dir, product, assets.get("creative") or "")
                if provided_asset_path:
                    self.logger.info(f"Using provided asset for {product}: {provided_asset_path}")

            add_logo = not assets['has_logo']
            add_message = not assets['has_message']

            for region in regions:
                
                report["checks"].setdefault(product, {})
                report["checks"][product].setdefault(region, {})
                report["checks"][product][region]["legal"] = True
                report["checks"][product][region]["brand"] = True

                base_prompt_core = ""

                if add_message:

                    # translate
                    message = en_message if region == 'en-US' else translate_message(en_message, region)
                    if region != 'en-US':
                        self.logger.info(f"Translated message for {region}: {message}")
                    _tick(report, self.output_dir, product, region, "translate", "done", {"message": message})
                    # legal
                    if not legal_check(message, region):
                        error = f"Legal check failed for {product} in {region}"
                        self.logger.error(error)
                        report["checks"][product][region]["legal"] = False
                        _tick(report, self.output_dir, product, region, "legal", "failed", {"error": error})
                        _set_overall_status(report, self.output_dir, "failed")
                        return {"status": "failed", "error": error, "report_path": f"{self.output_dir}/report.json"}
                    _tick(report, self.output_dir, product, region, "legal", "done")

                if provided_asset_path:
                    # Skip generation entirely; use provided 1:1 as the "hero"
                    hero_1x1_path = provided_asset_path 
                    _tick(
                        report, self.output_dir, product, region, "gen_1x1",
                        "done", {"path": hero_1x1_path, "source": "provided_asset", "note": "generation skipped"}
                    )
                else:

                    # resolve assets
                    input_image_path, mask_image_path = _resolve_product_asset_paths(product, region)
                    if not input_image_path:
                        asset_data = assets.get('creative') if isinstance(assets, dict) else None
                        input_image_path = _resolve_asset_from_base64_or_path(self.output_dir, product, region, asset_data or "")
                        if input_image_path:
                            self.logger.info(f"Using provided asset for {product} in {region}: {input_image_path}")

                    if not input_image_path:
                        error = (f"No input image for {product} in {region}. "
                                f"Looked for '{ASSETS_DIR}/{product}__{region}.png' or a valid asset in brief.")
                        self.logger.error(error)
                        report["checks"].setdefault(product, {})
                        report["checks"][product][f"{product}_{region}_asset"] = False  # >>> grouped-by-product
                        _tick(report, self.output_dir, product, region, "gen_1x1", "failed", {"error": error})
                        _set_overall_status(report, self.output_dir, "failed")
                        return {"status": "failed", "error": error, "report_path": f"{self.output_dir}/report.json"}

                    if not mask_image_path:
                        error = (f"No mask image for {product}. "
                                f"Expected '{ASSETS_DIR}/{product}__mask.png'.")
                        self.logger.error(error) 
                        _tick(report, self.output_dir, product, region, "gen_1x1", "failed", {"error": error})
                        _set_overall_status(report, self.output_dir, "failed")
                        return {"status": "failed", "error": error, "report_path": f"{self.output_dir}/report.json"}


                    # build prompt
                    base_prompt_core = generate_prompt(theme, product, region, audience, message)
                    base_prompt = (
                        f"Do not alter the protected regions. Edit background only. "
                        f"Lower 30% of the image must be clean, uniform, with minimal detail and low contrast design. "
                        f"The camera position, crop, and perspective of the existing can are correct. "
                        f"Generate background elements consistent with that same camera position: {base_prompt_core}. "
                        f"Keep the {product} can exactly the same size, position, and proportions as in the original — no scaling, cropping, or perspective change. *** You MUST leave enough negative space at the bottom (30% of the image height) to add text and logos later. ***"
                    )

                    

                    # gen 1:1
                    try:
                        hero_1x1_path = generate_image(
                            prompt=base_prompt,
                            product=product,
                            region=region,
                            output_dir=self.output_dir,
                            output_name=f"{product}_{region}_hero.png",
                            size="1024x1024",
                            input_image_path=input_image_path,
                            mask_image_path=mask_image_path  
                        )
                        self.logger.info(f"Generated 1:1 hero for {product} in {region}: {hero_1x1_path}")
                        _tick(report, self.output_dir, product, region, "gen_1x1", "done", {"path": hero_1x1_path})
                    except Exception as e:
                        error = f"Image generation failed for {product} in {region}: {e}"
                        self.logger.error(error)
                        _tick(report, self.output_dir, product, region, "gen_1x1", "failed", {"error": str(e)})
                        _set_overall_status(report, self.output_dir, "failed")
                        return {"status": "failed", "error": error, "report_path": f"{self.output_dir}/report.json"}

                # compose 1:1
                if "1:1" in aspect_ratios:
                    out_path = compose_creatives(
                        hero_1x1_path,
                        message if add_message else None,
                        "1:1",
                        logo_path if add_logo else None,
                        product,
                        region,
                        self.output_dir,
                        f"{product}_{region}_hero_1-1_ad.png"
                    )
                    # record at product level + per product/region
                    report["progress"]["per"][product][region]["outputs"].append(out_path)
                    _atomic_save_json(os.path.join(self.output_dir, "report.json"), report)
                    _tick(report, self.output_dir, product, region, "compose_1x1", "done", {"path": out_path})
                    _add_output(report, self.output_dir, product, out_path, region)

                ar_to_hero_path = {"1:1": hero_1x1_path}
                temp_dir = os.path.join(self.output_dir, "temp"); _ensure_dir(temp_dir)

                # derive 16:9
                if "16:9" in aspect_ratios:

                    try:

                        if not provided_asset_path or (provided_asset_path and  not base16_9):

                            seed = os.path.join(temp_dir, f"{product}_{region}_hero_16-9_seed.png")
                            _build_white_canvas_with_1x1(hero_1x1_path, seed, 1536, 1024, "right")
                            mask_169 = os.path.join(ASSETS_DIR, "mask_16-9.png")

                            if not os.path.exists(mask_169): raise FileNotFoundError(f"Missing mask file: {mask_169}")

                            hero_169_full = generate_image(
                                prompt= f"Replace the orange rectangle with a background that blends with the rest of the imge, retain everything else, do not change the position of the can. {base_prompt_core}",
                                product=product, region=region, output_dir=self.output_dir,
                                output_name=f"{product}_{region}_hero_16-9.png",
                                size="1536x1024", input_image_path=seed, 
                                # mask_image_path=mask_169
                            )
                            hero_169_cropped = hero_169_full
                            _crop_center_keep_size(hero_169_full, hero_169_cropped, final_w=None, final_h=864)
                            ar_to_hero_path["16:9"] = hero_169_cropped
                            _tick(report, self.output_dir, product, region, "derive_16_9", "done", {"path": hero_169_cropped})

                            base16_9 = hero_169_cropped  # cache for next regions
                        else:
                            hero_169_cropped = base16_9
                            _tick(report, self.output_dir, product, region, "derive_16_9", "done", {"path": hero_169_cropped, "source": "cached"})
                        

                        out_path = compose_creatives(
                            hero_169_cropped,
                            message if add_message else None,
                            "16:9",
                            logo_path if add_logo else None,
                            product, region, self.output_dir,
                            f"{product}_{region}_hero_16-9_ad.png"
                        )
                        report["progress"]["per"][product][region]["outputs"].append(out_path)
                        _atomic_save_json(os.path.join(self.output_dir, "report.json"), report)
                        _tick(report, self.output_dir, product, region, "compose_16_9", "done", {"path": out_path})
                        _add_output(report, self.output_dir, product, out_path, region)
                    except Exception as e:
                        self.logger.error(f"Failed 16:9 derivation for {product} in {region}: {e}")
                        _tick(report, self.output_dir, product, region, "derive_16_9", "failed", {"error": str(e)})

                # derive 9:16
                if "9:16" in aspect_ratios:
                    try:

                        if not provided_asset_path or (provided_asset_path and not base9_16):

                            seed = os.path.join(temp_dir, f"{product}_{region}_hero_9-16_seed.png")
                            _build_white_canvas_with_1x1(hero_1x1_path, seed, 1024, 1536, "top")
                            mask_916 = os.path.join(ASSETS_DIR, "mask_9-16.png")
                            if not os.path.exists(mask_916): raise FileNotFoundError(f"Missing mask file: {mask_916}")
                            hero_916_full = generate_image(
                                # prompt="Replace the orange rectangle with a background consistent with the existing background, retain everything else, do not change the position of the can",
                                prompt= f"Replace the orange rectangle with a background that blends with the rest of the imge, retain everything else, do not change the position of the can. {base_prompt_core}",
                                product=product, region=region, output_dir=self.output_dir,
                                output_name=f"{product}_{region}_hero_9-16.png",
                                size="1024x1536", input_image_path=seed, 
                                # mask_image_path=mask_916
                            )
                            hero_916_cropped = hero_916_full
                            _crop_center_keep_size(hero_916_full, hero_916_cropped, final_w=864, final_h=None)
                            ar_to_hero_path["9:16"] = hero_916_cropped
                            _tick(report, self.output_dir, product, region, "derive_9_16", "done", {"path": hero_916_cropped})

                            base9_16 = hero_916_cropped  # cache for next regions

                        else:
                            hero_916_cropped = base9_16
                            _tick(report, self.output_dir, product, region, "derive_9_16", "done", {"path": hero_916_cropped, "source": "cached"})

                        out_path = compose_creatives( 
                            hero_916_cropped,
                            message if add_message else None,
                            "9:16",
                            logo_path if add_logo else None,
                            product, region, self.output_dir,
                            f"{product}_{region}_hero_9-16_ad.png"
                        )
                        report["progress"]["per"][product][region]["outputs"].append(out_path)
                        _atomic_save_json(os.path.join(self.output_dir, "report.json"), report)
                        _tick(report, self.output_dir, product, region, "compose_9_16", "done", {"path": out_path})
                        _add_output(report, self.output_dir, product, out_path, region)
                    except Exception as e:
                        self.logger.error(f"Failed 9:16 derivation for {product} in {region}: {e}")
                        _tick(report, self.output_dir, product, region, "derive_9_16", "failed", {"error": str(e)})

                # brand compliance over only this product's outputs
                try:
                    for region, variants in report["outputs"].get(product, {}).items():
                        for variant_path in variants:  # it's a string now
                            if variant_path and os.path.basename(variant_path).startswith(product) and os.path.exists(variant_path):
                                ok = brand_compliance(variant_path, brand)
                                _add_compliance(report, self.output_dir, product, region, variant_path, ok)
                                if not ok:
                                    self.logger.warning(f"Compliance check failed for {variant_path}")
                except Exception as e:
                    self.logger.error(f"Failed brand check: {e}")

        _set_overall_status(report, self.output_dir, "complete")
        self.logger.info("Job complete")
        return {"status": report["status"], "report_path": f"{self.output_dir}/report.json"}

# to test directly
if __name__ == "__main__":
    sample_brief = {
        "id": "5745-sd585-981",
        "products": [
            {
                "id": "cocacola_zero_cherry",
                "regions": ["en-US", "es-ES", "fr-FR", "ja-JP"],
                "audience": "18-30 year olds, urban, active lifestyle",
                "message": "Stay sharp, stay cherry.",
                "theme": "vibrant, energetic, youthful",
                "aspect_ratios": ["1:1", "9:16", "16:9"],
                "assets": {"creative": "", "has_logo": False, "has_message": False}
            },
            {
                "id": "sprite_lemon_lime",
                "regions": ["en-US", "es-ES", "fr-FR", "ja-JP"],
                "audience": "30–50 year olds, balanced, health-conscious professionals seeking clarity and refreshment",
                "message": "Zest mode: on.",
                "theme": "fresh, clean, invigorating",
                "aspect_ratios": ["1:1", "9:16", "16:9"],
                "assets": {"creative": "base64_img_data_here", "has_logo": False, "has_message": False}
            }
        ]
    }
    manager = CampaignManager(str(uuid4()))
    print(manager.process(sample_brief))
