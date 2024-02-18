from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

# Video dosyası ve model yolunu belirtme
source ="uk.mp4"
model_path = 'yolov8n.pt'

# YOLO modelini yükleme
model = YOLO(model_path)

# Video dosyasını açma
cap = cv2.VideoCapture(source)
assert cap.isOpened(), "Error reading video file"

frame_count = 0


# Video akışı devam ettiği sürece döngüde kal
while cap.isOpened():
    # Bir sonraki frame'i oku
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = model.predict(frame,stream=True)
    names = model.names
    #Sınıfların isimlerini ve tespit sayılarını tutacak sözlük
    class_counts = {}
    for r in results:
        for c in r.boxes.cls:
            # c sınıfın içerisinde bulduğu objenin numarasıdır c = 0 aslında person dır.
            class_name = names[int(c)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
   # Frame numarası ve tespit sayılarını dosyaya yaz
    with open("object_frames.txt", "a") as output_file:
        output_file.write(f"Frame {frame_count}:\n")
        for class_name, count in class_counts.items():
            output_file.write(f"{class_name}: {count}\n")
    
    # Frame sayısını arttır
  
    frame_count += 1

# Kaynakları serbest bırakma
cap.release()
output_file.close()
cv2.destroyAllWindows()

