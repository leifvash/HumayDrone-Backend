"""
Test and Ingest Raw DJI Drone Flight Images.
Automatically extracts GPS EXIF metadata (Latitude, Longitude, Altitude, Capture Time),
runs 4-stage Neural Cloud Inference (YOLOv8 + U-Net + ResNet-50), and optionally
uploads to the Live Render Cloud Backend.
"""
import os
import sys
import argparse
import glob
from pathlib import Path
from PIL import Image, ExifTags
import requests

# Ensure cloud-backend is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from inference.pipeline import get_inference_pipeline


def extract_dji_exif_gps(image_path: str):
    """
    Extracts real Latitude, Longitude, RelativeAltitude (AGL), and Timestamp from DJI XMP + EXIF tags.
    Note: Standard EXIF GPSAltitude (GPSInfo[6]) on DJI drones stores AbsoluteAltitude (ASL),
    whereas XMP `drone-dji:RelativeAltitude` stores the true height above ground level (AGL).
    """
    import re
    rel_alt = None
    try:
        with open(image_path, "rb") as f:
            head = f.read(65536)
        match = re.search(rb'drone-dji:RelativeAltitude="([+-]?\d+(?:\.\d+)?)"', head)
        if match:
            rel_alt = round(abs(float(match.group(1).decode("ascii"))), 2)
    except Exception:
        pass

    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None, None, rel_alt, None

            gps_info = None
            capture_dt = None

            for tag, val in exif.items():
                tag_name = ExifTags.TAGS.get(tag, str(tag))
                if tag_name == "GPSInfo":
                    gps_info = val
                elif tag_name in ("DateTimeOriginal", "DateTime"):
                    capture_dt = str(val)

            if not gps_info:
                return None, None, rel_alt, capture_dt

            # GPS Latitude & Longitude conversion from degrees, minutes, seconds
            def dms_to_dd(dms, ref):
                degrees = float(dms[0])
                minutes = float(dms[1])
                seconds = float(dms[2])
                dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
                if ref in ("S", "W"):
                    dd = -dd
                return dd

            lat = None
            lng = None
            alt = rel_alt

            if 2 in gps_info and 1 in gps_info:
                lat = round(dms_to_dd(gps_info[2], gps_info[1]), 7)
            if 4 in gps_info and 3 in gps_info:
                lng = round(dms_to_dd(gps_info[4], gps_info[3]), 7)
            if alt is None and 6 in gps_info:
                alt = round(float(gps_info[6]), 2)

            return lat, lng, alt, capture_dt
    except Exception:
        return None, None, rel_alt, None


def run_local_inference(image_paths, max_images=10):
    """
    Executes inference locally using bundled YOLOv8, UNet, and ResNet-50 weights.
    """
    pipeline = get_inference_pipeline()
    print("\n" + "=" * 78)
    print(f"  HUMAYDRONE LOCAL INFERENCE: PROCESSING {len(image_paths)} DJI DRONE IMAGES")
    print("=" * 78)

    results = []
    for idx, img_path in enumerate(image_paths[:max_images], 1):
        filename = os.path.basename(img_path)
        lat, lng, alt, dt = extract_dji_exif_gps(img_path)
        gps_str = f"{lat:.6f}N, {lng:.6f}E (Alt: {alt}m)" if lat and lng else "No GPS in EXIF"

        print(f"\n[{idx}/{min(len(image_paths), max_images)}] Analyzing: {filename}")
        print(f"  GPS Coordinate : {gps_str}")
        if dt:
            print(f"  Captured Time  : {dt}")

        res = pipeline.process_image(img_path)
        cls_label = res.get("classification_label", "Unknown")
        conf = res.get("confidence_score", 0.0) * 100
        sev = res.get("severity_score", 0.0)
        mask = res.get("mask_relative_path", "")

        print(f"  Diagnosis      : {cls_label} (Confidence: {conf:.1f}%)")
        print(f"  Canopy Severity: {sev:.2f}% discoloration")
        print(f"  Treatment      : {res.get('treatment_recommendation', 'N/A')}")
        if mask:
            print(f"  Generated Mask : {mask}")

        results.append({
            "filename": filename,
            "lat": lat,
            "lng": lng,
            "alt": alt,
            "classification": cls_label,
            "confidence": conf,
            "severity": sev,
        })

    print("\n" + "=" * 78)
    print("  FLIGHT SUMMARY")
    print("=" * 78)
    for r in results:
        gps = f"({r['lat']:.5f}, {r['lng']:.5f})" if r['lat'] else "(--, --)"
        print(f"  * {r['filename']:<14} | {gps:<22} | {r['classification']:<22} | Conf: {r['confidence']:5.1f}% | Sev: {r['severity']:5.2f}%")
    print("=" * 78)


def run_cloud_upload(image_paths, backend_url, flight_title="DJI Flight Mission", max_images=10):
    """
    Uploads images and extracted GPS coordinates to the live Render / cloud backend.
    """
    print("\n" + "=" * 78)
    print(f"  UPLOADING TO LIVE CLOUD BACKEND: {backend_url}")
    print(f"  Flight Title: {flight_title}")
    print("=" * 78)

    # 1. Create a flight session
    try:
        flight_res = requests.post(
            f"{backend_url}/flights/",
            json={"title": flight_title, "notes": f"Ingested {len(image_paths)} DJI Fly App captures."},
            timeout=30
        )
        if flight_res.status_code not in (200, 201):
            print(f"[!] Failed to create flight on backend: {flight_res.text}")
            return
        flight_id = flight_res.json().get("id")
        print(f"[OK] Created Cloud Flight Session ID #{flight_id}")
    except Exception as e:
        print(f"[X] Connection error connecting to {backend_url}: {e}")
        return

    # 2. Upload images with GPS coordinates
    for idx, img_path in enumerate(image_paths[:max_images], 1):
        filename = os.path.basename(img_path)
        lat, lng, alt, _ = extract_dji_exif_gps(img_path)

        data = {
            "flight_id": flight_id,
            "latitude": lat if lat is not None else 8.5044,
            "longitude": lng if lng is not None else 124.5759,
            "altitude": alt if alt is not None else 3.0,
        }

        with open(img_path, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}
            try:
                up_res = requests.post(f"{backend_url}/images/upload/", data=data, files=files, timeout=60)
                if up_res.status_code in (200, 201, 202):
                    res_json = up_res.json()
                    img_id = res_json.get("image_id")
                    print(f"  [{idx}/{min(len(image_paths), max_images)}] Uploaded {filename} -> Image ID #{img_id} (GPS: {data['latitude']:.5f}, {data['longitude']:.5f})")
                else:
                    print(f"  [{idx}] Upload failed for {filename}: {up_res.text}")
            except Exception as e:
                print(f"  [{idx}] Error uploading {filename}: {e}")

    # 3. Retrieve final flight analytics report
    try:
        import time
        time.sleep(3)
        rep_res = requests.get(f"{backend_url}/flights/{flight_id}/report/", timeout=30)
        if rep_res.status_code == 200:
            rep = rep_res.json()
            print("\n" + "=" * 78)
            print("  LIVE CLOUD ANALYTICS & GEOJSON MAP REPORT")
            print("=" * 78)
            print(f"  Total Captures Analyzed : {rep.get('total_captures', 0)}")
            print(f"  Average Discoloration   : {rep.get('average_discoloration_severity', 0.0):.2f}%")
            print(f"  Breakdown by Disease    : {rep.get('disease_breakdown', {})}")
            print(f"  GeoJSON Features Mapped : {len(rep.get('geojson_heatmap', {}).get('features', []))} points")
            print(f"  View Live GeoJSON Report: {backend_url}/flights/{flight_id}/report/")
            print("=" * 78)
    except Exception as e:
        print(f"Report fetch warning: {e}")


def main():
    parser = argparse.ArgumentParser(description="HumayDrone Raw Drone Image Testing Tool")
    parser.add_argument("--dir", type=str, required=True, help="Path to folder containing DJI drone images")
    parser.add_argument("--count", type=int, default=5, help="Number of images to process (default: 5)")
    parser.add_argument("--upload", action="store_true", help="Upload to the live Render cloud backend")
    parser.add_argument("--backend", type=str, default="https://humaydrone-backend.onrender.com/api", help="Backend API URL")
    parser.add_argument("--title", type=str, default="DJI Batch Flight Mission", help="Flight session title")
    args = parser.parse_args()

    target_dir = os.path.abspath(args.dir)
    if not os.path.exists(target_dir):
        print(f"[X] Directory not found: {target_dir}")
        sys.exit(1)

    # Collect images
    extensions = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.png", "*.PNG")
    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(target_dir, ext)))
    all_images = sorted(list(set(all_images)))

    if not all_images:
        print(f"[X] No images found in: {target_dir}")
        sys.exit(1)

    print(f"[*] Found {len(all_images)} drone captures in: {target_dir}")

    # Run local inference
    run_local_inference(all_images, max_images=args.count)

    # Optionally upload to cloud
    if args.upload:
        run_cloud_upload(all_images, backend_url=args.backend, flight_title=args.title, max_images=args.count)


if __name__ == "__main__":
    main()
