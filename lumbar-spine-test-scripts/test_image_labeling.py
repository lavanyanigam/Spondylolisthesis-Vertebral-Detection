import os
import csv
from PIL import Image


src_dir = "/Users/lavanyanigam/Desktop/spondylolisthesis-vertebral-project/lumbar-spine-clean-test-images"
csv_path = "/Users/lavanyanigam/Desktop/spondylolisthesis-vertebral-project/lumbar-spine-test-labels/lumbar-spine-test-label-actual.csv"


os.makedirs(os.path.dirname(csv_path), exist_ok=True)


with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    
    
    writer.writerow(["Image_Name", "Diagnosis"])
    
    
    img_files = sorted(os.listdir(src_dir))

    for img in img_files:
        
        if img.lower().endswith(".jpg"):
            act_path = os.path.join(src_dir, img)
            
            try:
                
                img_obj = Image.open(act_path)
                img_obj.show()
                
                
                diagnosis = input(f"Looking at {img}. Enter diagnosis (or type 'q' to quit): ")
                
                if diagnosis.lower() == 'q':
                    print("Quitting labeling process...")
                    break
                
                
                writer.writerow([img, diagnosis])
                
            except Exception as e:
                print(f"Could not open {img}. Error: {e}")

print("done")