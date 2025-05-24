import os
from PIL import Image

input_labels_dir = 'VisDrone2019-DET-train/annotations'
input_images_dir = 'VisDrone2019-DET-train/images'
output_labels_dir = 'labels_yolo'

os.makedirs(output_labels_dir, exist_ok=True)

for label_file in os.listdir(input_labels_dir):
    if not label_file.endswith('.txt'):
        continue

    label_path = os.path.join(input_labels_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(input_images_dir, image_name)
    output_path = os.path.join(output_labels_dir, label_file)

    if not os.path.exists(image_path):
        print(f"⚠️ Obraz nie istnieje: {image_path}")
        continue

    with Image.open(image_path) as img:
        width, height = img.size

    with open(label_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue

            x, y, w, h, category = map(int, parts[:5])

            if category <= 0:
                continue  # Pomiń kategorię 0 (lub -1 po konwersji)

            if int(parts[6]) == 1:
                continue  # truncated = 1 → pomiń

            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            # Zabezpieczenie przed out-of-bounds
            x_center = min(x_center, 1.0)
            y_center = min(y_center, 1.0)
            w_norm = min(w_norm, 1.0)
            h_norm = min(h_norm, 1.0)

            class_id = category - 1

            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ Konwersja zakończona.")
