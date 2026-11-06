import os
import sys
import glob
import time
import requests
import json

BASE_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000/api")

def run_test(image_folder_or_file: str, flight_title: str = "Test Flight - Disease Dataset"):
    print(f"\n=======================================================")
    print(f"  HUMAYDRONE CLOUD INGESTION & INFERENCE TEST")
    print(f"=======================================================")
    print(f"Target Server : {BASE_URL}")
    print(f"Input Target  : {image_folder_or_file}")
    
    # Check server availability
    try:
        r = requests.get(f"{BASE_URL}/flights/")
        if r.status_code != 200:
            print(f"[!] Server responded with status {r.status_code}. Is it running?")
            return
    except requests.exceptions.ConnectionError:
        print(f"[X] Could not connect to {BASE_URL}!")
        print("    Please start the server first: python manage.py runserver 0.0.0.0:8000")
        return

    # Find image files
    if os.path.isdir(image_folder_or_file):
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_folder_or_file, "**", ext), recursive=True))
        image_files = sorted(list(set(image_files)))

        # If there are many images, select a balanced sample across classes
        if len(image_files) > 15:
            blight_imgs = [p for p in image_files if "blight" in p.lower() or "bacterial" in p.lower()][:3]
            whitehead_imgs = [p for p in image_files if "white_head" in p.lower() or "whitehead" in p.lower()][:3]
            healthy_imgs = [p for p in image_files if "healthy" in p.lower()][:3]
            deadheart_imgs = [p for p in image_files if "dead" in p.lower() or "heart" in p.lower()][:3]
            selected = blight_imgs + whitehead_imgs + healthy_imgs + deadheart_imgs
            if selected:
                print(f"[*] Selecting balanced sample of {len(selected)} images across available classes.")
                image_files = selected
    elif os.path.isfile(image_folder_or_file):
        image_files = [image_folder_or_file]
    else:
        print(f"[X] Path does not exist: {image_folder_or_file}")
        return

    if not image_files:
        print(f"[!] No image files found in {image_folder_or_file}")
        return

    print(f"[OK] Found {len(image_files)} image(s) to process.")

    # 1. Create a Test Flight
    print(f"\n[1/3] Creating DroneFlight session...")
    flight_resp = requests.post(f"{BASE_URL}/flights/", json={
        "title": flight_title,
        "flight_path": [
            {"lat": 8.45420, "lon": 124.63190},
            {"lat": 8.45465, "lon": 124.63235}
        ]
    })
    if flight_resp.status_code not in (200, 201):
        print(f"[X] Failed to create flight: {flight_resp.text}")
        return
    
    flight_data = flight_resp.json()
    flight_id = flight_data.get("id")
    print(f"[OK] Created Flight ID #{flight_id}: '{flight_title}'")

    # 2. Upload images one by one with simulated GPS grid coordinates
    print(f"\n[2/3] Uploading images to /api/images/upload/ ...")
    base_lat = 8.45420
    base_lon = 124.63190
    step = 0.00010 # roughly 10 meters

    uploaded_results = []

    for idx, img_path in enumerate(image_files[:15]): # Test up to 15 images
        lat = base_lat + (idx % 4) * step
        lon = base_lon + (idx // 4) * step
        filename = os.path.basename(img_path)

        # Detect label hint from parent folder or filename
        lower_name = (img_path).lower()
        if "bacterial" in lower_name or "blb" in lower_name or "blight" in lower_name:
            label_hint = "Bacterial Leaf Blight"
        elif "dead" in lower_name or "heart" in lower_name:
            label_hint = "Dead Heart"
        elif "white" in lower_name or "head" in lower_name:
            label_hint = "White Head"
        elif "healthy" in lower_name:
            label_hint = "Healthy"
        else:
            label_hint = "Bacterial Leaf Blight"

        edge_meta = json.dumps([{
            "label": label_hint,
            "confidence": 0.88,
            "model_used": "YOLOv8-Tiny"
        }])

        with open(img_path, 'rb') as f:
            files = {'file': (filename, f, 'image/jpeg')}
            data = {
                'flight_id': flight_id,
                'latitude': lat,
                'longitude': lon,
                'altitude': 3.5,
                'edge_detections': edge_meta
            }
            res = requests.post(f"{BASE_URL}/images/upload/", data=data, files=files)

        if res.status_code == 202:
            resp_json = res.json()
            image_id = resp_json.get("image_id")
            task_id = resp_json.get("task_id")
            print(f"  [+] Uploaded ({idx+1}/{min(len(image_files), 15)}): {filename} -> Image #{image_id} (Task: {task_id[:8]}...)")
            uploaded_results.append(image_id)
        else:
            print(f"  [X] Failed upload for {filename}: HTTP {res.status_code} - {res.text}")

    # Small pause for background tasks if Celery is running async
    time.sleep(1)

    # 3. Fetch analytics report
    print(f"\n[3/3] Fetching Generated Flight Analytics Report...")
    report_resp = requests.get(f"{BASE_URL}/flights/{flight_id}/report/")
    if report_resp.status_code == 200:
        report = report_resp.json()
        stats = report.get("summary_statistics", {})
        avg_sev = stats.get("average_severity_percentage", stats.get("average_severity_score", 0))
        breakdown = stats.get("disease_breakdown", stats.get("disease_counts", {}))
        
        print(f"\n=======================================================")
        print(f"                  ANALYTICS SUMMARY                    ")
        print(f"=======================================================")
        print(f"Total Captures Analyzed : {stats.get('total_images_analyzed', 0)}")
        print(f"Average Discoloration   : {avg_sev:.2f}%")
        print(f"Class Breakdown         : {json.dumps(breakdown, indent=4)}")
        
        recs = report.get("recommendations", "")
        if recs:
            print(f"\n[Advisory Recommendations]")
            print(f"  {recs}")
        
        geojson = report.get("severity_map", {})
        features = geojson.get("features", [])
        print(f"\nGeoJSON Heatmap Pinpoints Generated : {len(features)} points")
        for f in features:
            props = f.get("properties", {})
            mask_str = f" | Mask: {props.get('segmentation_mask')}" if props.get('segmentation_mask') else ""
            print(f"  * Image #{props.get('image_id')}: {props.get('disease')} | Severity: {props.get('severity'):.2f}% | Conf: {props.get('confidence')*100:.1f}%{mask_str}")
        print(f"GeoJSON Report Endpoint : {BASE_URL}/flights/{flight_id}/report/")
        print(f"=======================================================\n")
    else:
        print(f"[!] Report not ready yet or error: HTTP {report_resp.status_code}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        # Default test image if no arg provided
        target = r"C:\Users\Aaron\StudioProjects\HumayDrone\SampleCode-V5\android-sdk-v5-sample\src\main\res\drawable\login_hero_photo.jpg"
    run_test(target)
