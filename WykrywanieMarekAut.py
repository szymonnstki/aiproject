import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn


class CarBrandClassifier:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)

        # Wczytanie nazw klas
        self.class_names = checkpoint['class_names']
        print(f"Zarejestrowane marki aut: {self.class_names}")

        # Inicjalizacja modelu z odpowiednią architekturą
        self.model = models.resnet18(pretrained=False)

        # Dostosowanie warstw FC do struktury z checkpointu
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, len(self.class_names))
        )

        # Wczytanie wag
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()

        # Transformacje obrazu
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def classify(self, image):
        """Klasyfikacja marki auta na podstawie obrazu"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            return self.class_names[predicted.item()]


class CarDetectionSystem:
    def __init__(self, yolo_model_path, classifier_model_path):
        # Inicjalizacja detektora YOLOv5
        self.yolo_model = self.load_yolo_model(yolo_model_path)

        # Inicjalizacja klasyfikatora marek
        self.classifier = CarBrandClassifier(classifier_model_path)

    def load_yolo_model(self, model_path):
        """Ładowanie modelu YOLOv5 z obsługą błędów"""
        try:
            # Load model with force_reload and skip validation
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path=model_path,
                                   force_reload=True,
                                   skip_validation=True,
                                   verbose=False)
            print("Model YOLO pomyślnie załadowany")
            return model
        except Exception as e:
            print(f"Błąd ładowania modelu YOLO: {e}")
            print("Próba alternatywnego ładowania...")

            try:
                # Try loading with newer method
                from ultralytics import YOLO
                model = YOLO(model_path)
                return model
            except Exception as e2:
                print(f"Błąd alternatywnego ładowania: {e2}")
                raise RuntimeError("Nie można załadować modelu YOLO")

    def process_frame(self, frame, conf_threshold=0.75):
        """Przetwarzanie pojedynczej klatki wideo"""
        detections = []

        # Handle different YOLO versions
        if hasattr(self.yolo_model, 'predict'):
            # For newer YOLO versions (v8+)
            results = self.yolo_model.predict(frame)

            # Check if results is a list (YOLOv8) or has xyxy attribute (older YOLOv5)
            if isinstance(results, list):  # YOLOv8 format
                for result in results:
                    if hasattr(result, 'boxes'):
                        boxes = result.boxes
                        for box in boxes:
                            if result.names[int(box.cls)] == 'car' and box.conf > conf_threshold:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                detections.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': float(box.conf)
                                })
            else:  # Older YOLOv5 format
                for *box, conf, cls in results.xyxy[0]:
                    if results.names[int(cls)] == 'car' and conf > conf_threshold:
                        x1, y1, x2, y2 = map(int, box)
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': float(conf)
                        })
        else:
            # For older YOLOv5 versions
            results = self.yolo_model(frame)
            for *box, conf, cls in results.xyxy[0]:
                if self.yolo_model.names[int(cls)] == 'car' and conf > conf_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(conf)
                    })

        # Klasyfikacja i wizualizacja
        results = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            car_img = frame[y1:y2, x1:x2]

            if car_img.size == 0:
                continue

            try:
                brand = self.classifier.classify(car_img)
            except Exception as e:
                print(f"Błąd klasyfikacji: {e}")
                brand = "unknown"

            results.append({
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'brand': brand
            })

            # Rysowanie wyników tylko jeśli pewność ≥75%
            if det['confidence'] >= 0.75:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{brand} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return results, frame

def process_video(video_path, output_path, yolo_model_path, classifier_model_path):
    # Inicjalizacja systemu
    print("Inicjalizacja systemu detekcji i klasyfikacji aut...")
    system = CarDetectionSystem(yolo_model_path, classifier_model_path)

    # Otwarcie pliku wideo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć pliku wideo")
        return

    # Pobranie właściwości wideo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Inicjalizacja writer dla output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Przetwarzanie klatki {frame_count}...")

        # Przetwarzanie klatki
        results, processed_frame = system.process_frame(frame)

        # Zapis klatki do output video
        out.write(processed_frame)

        # Wyświetlenie podglądu
        cv2.imshow('Wynik', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zwolnienie zasobów
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Przetwarzanie wideo zakończone")

if __name__ == '__main__':
    # Konfiguracja ścieżek
    yolo_path = 'C:/Users/hachi/aiproject/runs/train/car-detection/weights/best.pt'
    classifier_path = 'C:/Users/hachi/aiproject/best_model_copy.pth'
    video_path = 'C:/Users/hachi/aiproject/dataimg/highway4.mp4'
    output_video_path = 'C:/Users/hachi/aiproject/output_video.mp4'

    # Przetwarzanie wideo
    process_video(video_path, output_video_path, yolo_path, classifier_path)