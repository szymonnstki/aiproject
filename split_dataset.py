import os
from PIL import Image

# Ścieżki
input_labels_dir = 'labels_visdrone'  # oryginalne pliki .txt
input_images_dir = 'images'           # folder z oryginalnymi obrazami
output_labels_dir = 'labels_yolo'     # folder wyjściowy z etykietami YOLO

# Utwórz folder wyjściowy, jeśli nie istnieje
os.makedirs(output_labels_dir, exist_ok=True)

# Przetwórz każdy plik etykiet
for label_file in os.listdir(input_labels_dir):
    if not label_file.endswith('.txt'):
        continue

    # Ścieżki
    label_path = os.path.join(input_labels_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(input_images_dir, image_name)
    output_path = os.path.join(output_labels_dir, label_file)

    # Sprawdź, czy odpowiadający obraz istnieje
    if not os.path.exists(image_path):
        print(f"⚠️ Obraz nie istnieje: {image_path}")
        continue

    # Pobierz rozdzielczość obrazu
    with Image.open(image_path) as img:
        width, height = img.size

    # Wczytaj etykiety i przelicz
    with open(label_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue

            x, y, w, h, category = map(int, parts[:5])

            # Pomiń obiekty ignore (opcjonalnie)
            if int(parts[6]) == 1:  # truncated=1 → ignorujemy
                continue

            # Konwersja do YOLO format
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            class_id = category - 1  # YOLO zaczyna od 0

            # Zapisz nowy wiersz
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("✅ Konwersja zakończona.")
