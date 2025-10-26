from ultralytics import YOLO
import os

# 1️⃣ 모델 경로 (백슬래시 주의!)
model = YOLO("runs/train/yolov8n_custom/weights/best.pt")

# 2️⃣ 이미지 폴더
img_dir = "test/images"  # 이미지 폴더 경로
output_csv = "predictions.csv"

# 3️⃣ 이미지 리스트
img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# 4️⃣ 예측
results = model.predict(img_files, imgsz=640, conf=0.65, verbose=False)

# 5️⃣ 결과 작성
lines = []
for r in results:
    img_name = os.path.splitext(os.path.basename(r.path))[0]  # 확장자 제거
    boxes = r.boxes

    if boxes is None or len(boxes) == 0:
        lines.append(f"{img_name},no box")
        continue

    parts = []
    for box in boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # YOLO 포맷과 비슷하게 center x,y,w,h 계산
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2

        # class conf cx cy w h
        parts.append(f"{cls} {conf:.3f} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f}")

    line = f"{img_name}," + " ".join(parts)
    lines.append(line)

# 6️⃣ 저장
with open(output_csv, "w", encoding="utf-8") as f:
    for line in lines:
        f.write(line + "\n")

print(f"✅ 완료! 결과가 '{output_csv}'에 저장되었습니다.")
