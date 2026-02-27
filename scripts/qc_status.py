import cv2
import pandas as pd
import numpy as np
import os 

raw_dir = "data/raw/"
subfolders = [f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))]
print(subfolders)
processed_dir = "data/processed"

# Creating the directory path specified in processed_dir
os.makedirs(processed_dir, exist_ok=True) 


# Stating the thresholds
blue_channel_min, blue_channel_max = 0.1, 0.6
exposure_min, exposure_max = 0.30, 0.90
resolution_val_min, resolution_val_max = (350,350), (12000,12000)
blur_threshold = 5
saturation_min = 0.10
occupancy_score_min, occupancy_score_max = 0.30, 0.70
rows = []

for folder_id in subfolders:
    folder_path = os.path.join(raw_dir, folder_id)
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        # Check if the file is corrupt
        if img is None:
            print(f"{filename} Failed, corrupt image detected")
            continue

        height, width, channels = img.shape

        # Resolution
        resolution_valid = (
            resolution_val_min[0] <= width <= resolution_val_max[0] and 
            resolution_val_min[1] <= height <= resolution_val_max[1] 
        )

        # Normalizing the image pixel data to be 0-1
        img_norm = img / 255.0 
    
        # Blue channel dominance
        blue_channel_score = img_norm[:, :, 0].mean()

        # Exposure
        exposure_score = img_norm.mean()

        # Saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
        saturation_score = (hsv[:, :, 1] / 255.0).mean() 

        # Occupancy
        h, s, v = cv2.split(hsv)

        s_norm = s / 255.0
        v_norm = v / 255.0

        color_mask = s_norm > 0.15 

        brightness_mask = v_norm > 0.05

        fruit_mask = color_mask & brightness_mask

        fruit_mask = fruit_mask.astype("uint8") * 255

        kernel = np.ones((7,7), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clean_mask = np.zeros_like(fruit_mask)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=-1)

        occupancy_score = np.sum(clean_mask > 0) / clean_mask.size


        # Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_valid = blur_score > blur_threshold # Higher blur score represents a sharper image
  
        qc_checks = {
            "blue_channel_issue": blue_channel_min <= blue_channel_score <= blue_channel_max,
            "exposure_issue": exposure_min <= exposure_score <= exposure_max,
            "resolution_issue": resolution_valid,
            "blur_issue": blur_valid,
            "saturation_issue": saturation_score >= saturation_min,
            "occupancy_issue": occupancy_score_min <= occupancy_score <= occupancy_score_max,

            }
    
        reasons = []
        if not (blue_channel_min <= blue_channel_score <= blue_channel_max): reasons.append("blue_channel_issue")
        if not (exposure_min <= exposure_score <= exposure_max): reasons.append("exposure_issue")
        if not (resolution_valid): reasons.append("resolution_issue")
        if not (blur_valid): reasons.append("blur_issue")
        if not (saturation_score >= saturation_min): reasons.append("saturation_issue")
        if not (occupancy_score_min <= occupancy_score <= occupancy_score_max): reasons.append("occupancy_issue")

        qc_status = "pass" if not reasons else "flagged"
    
        rows.append({
            "image_id": filename,
            "qc_status": qc_status,
            "blue_score": blue_channel_score,
            "exposure_score": exposure_score,
            "blur_score": blur_score,
            "saturation_score": saturation_score,
            "occupancy_score" : occupancy_score,
            "width": width,
            "height": height,
            "fail_reasons": ", ".join(reasons) if reasons else "none"
        })

qc = pd.DataFrame(rows)
qc.to_csv("labels/qc_metrics.csv", index=False)
print(qc["qc_status"].value_counts())

