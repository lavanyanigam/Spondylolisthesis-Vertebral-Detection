import os
import shutil


src_dir = "/Users/lavanyanigam/Desktop/spondylolisthesis-vertebral-project/lumbar-spine-test-images"
clean_path = "/Users/lavanyanigam/Desktop/spondylolisthesis-vertebral-project/lumbar-spine-test-labels/lumbar-spine-test-label-actual.csv"

os.makedirs(clean_path, exist_ok=True)

img_files = os.listdir(src_dir)

count = 0
for img in img_files:
    
    if img.endswith(".JPG"):
        count += 1
        clean_img = "test" + str(count) + ".JPG"
        
        
        act_path = os.path.join(src_dir, img)
        imgpth = os.path.join(clean_path, clean_img)
        
        
        if not os.path.isfile(imgpth):
            shutil.copy(act_path, imgpth)
            print(f"copied {img} as {clean_img}")
        else:
            print(f"{clean_img} skipped")

print("done")