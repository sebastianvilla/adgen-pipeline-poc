# pip install opencv-python numpy
import cv2
import numpy as np
import os

BRANDS = {
    "cocacola": {"logo": "assets/cocacola_logo.png"},
    "sprite":   {"logo": "assets/sprite_logo.png"},
}

def _load_logo_rgb(path: str):
    """Load logo, trim transparent padding, return 3-ch BGR."""
    tpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if tpl is None:
        raise ValueError(f"Logo not found: {path}")
    if tpl.ndim == 3 and tpl.shape[2] == 4:
        alpha = tpl[:, :, 3]
        ys, xs = np.where(alpha > 10)
        if ys.size == 0 or xs.size == 0:
            raise ValueError("Logo has no visible pixels")
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        tpl = tpl[y0:y1, x0:x1, :3]
    else:
        tpl = tpl[:, :, :3]
    return tpl

def _ncc(a, b):
    """Normalized cross-correlation in [-1,1]."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    a -= a.mean(); b -= b.mean()
    denom = (a.std() * b.std()) + 1e-6
    return float((a * b).mean() / denom)

def check(image_path: str, product: str, *, debug: bool = True) -> bool:
    """
    Return True iff the brand's logo is present in the image.
    Robust to size/rotation/position. No assumptions about placement.
    """
    if product not in BRANDS:
        return False
    if not os.path.exists(image_path):
        raise ValueError(f"Missing image: {image_path}")

    # --- load & prep ---
    img0 = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img0 is None:
        raise ValueError("Failed to read image")

    tpl0 = _load_logo_rgb(BRANDS[product]["logo"])

    # Upscale once so small logos yield features (deterministic)
    UPSCALE = 2.0
    img = cv2.resize(img0, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl_g = cv2.cvtColor(tpl0, cv2.COLOR_BGR2GRAY)

    # --- features ---
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01, edgeThreshold=8)
    kpI, desI = sift.detectAndCompute(img_g, None)
    kpT, desT = sift.detectAndCompute(tpl_g, None)
    if desI is None or desT is None or len(kpI) < 8 or len(kpT) < 8:
        if debug: print("Not enough features.")
        return False

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = flann.knnMatch(desT, desI, k=2)
    good = [m for m, n in knn if m.distance < 0.72 * n.distance]
    if debug: print("good matches:", len(good))
    if len(good) < 12:
        return False

    ptsT = np.float32([kpT[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsI = np.float32([kpI[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # --- robust geometry (USAC > RANSAC) ---
    H, mask = cv2.findHomography(ptsT, ptsI, method=cv2.USAC_MAGSAC, ransacReprojThreshold=3.0)
    if H is None or mask is None:
        if debug: print("No homography.")
        return False

    inliers = int(mask.sum())
    if debug: print("inliers:", inliers)
    if inliers < 10:
        return False

    # reprojection error on inliers
    proj = cv2.perspectiveTransform(ptsT, H)
    err = np.linalg.norm((proj - ptsI).reshape(-1, 2), axis=1)
    med_err = float(np.median(err[mask.ravel() == 1]))
    if debug: print("median reprojection error:", med_err)
    if med_err > 3.0:
        return False

    # geometry sanity (avoid degenerate fits)
    hI, wI = img.shape[:2]
    hT, wT = tpl0.shape[:2]
    corners = np.float32([[0, 0], [wT, 0], [wT, hT], [0, hT]]).reshape(-1, 1, 2)
    quad = cv2.perspectiveTransform(corners, H).reshape(4, 2).astype(np.float32)
    area = cv2.contourArea(quad)
    frac = area / (wI * hI + 1e-6)
    if debug: print("area fraction:", frac)
    if frac < 0.00008 or frac > 0.50:
        return False

    # light content check (template vs detected patch)
    x, y, w, h = cv2.boundingRect(quad)
    if w < 10 or h < 10:
        return False
    warped = cv2.warpPerspective(tpl_g, H, (wI, hI),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ncc = _ncc(img_g[y:y+h, x:x+w], warped[y:y+h, x:x+w])
    if debug: print("ncc:", ncc)
    if ncc < 0.58:
        return False

    return True
