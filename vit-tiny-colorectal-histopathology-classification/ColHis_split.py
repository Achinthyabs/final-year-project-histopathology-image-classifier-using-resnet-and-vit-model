import os
import cv2

# INPUT dataset (your raw dataset)
source_dir = "ColHis-IDS"

# OUTPUT dataset (clean dataset)
output_dir = "Processed_Dataset"

img_size = 224

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    for patient_folder in os.listdir(class_path):
        patient_path = os.path.join(class_path, patient_folder)

        if not os.path.isdir(patient_path):
            continue

        # Go inside zoom folders
        zoom_200_path = os.path.join(patient_path, "200")

        if not os.path.exists(zoom_200_path):
            continue  # skip if 200x not available

        # Create output class folder
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        for img_name in os.listdir(zoom_200_path):
            img_path = os.path.join(zoom_200_path, img_name)

            try:
                img = cv2.imread(img_path)

                if img is None:
                    continue

                # Resize to 224x224
                img_resized = cv2.resize(img, (img_size, img_size))

                # Save image
                save_path = os.path.join(output_class_dir, img_name)
                cv2.imwrite(save_path, img_resized)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

print("Dataset prepared successfully!")