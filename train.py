from ultralytics import YOLO

if __name__ == "__main__" :
    model = YOLO("yolov8n.pt")
    model.info()

    model.train(
        data="dataset.yaml",        # 데이터셋 경로 (train/val 이미지와 클래스 포함)
        epochs=100,              # 학습 epoch 수
        imgsz=640,               # 입력 이미지 크기
        batch=16,                # 배치 사이즈
        device=0,                # GPU 번호 (ex: 0 or '0,1' or 'cpu')
        optimizer="AdamW",       # SGD, Adam, AdamW 등 가능
        lr0=1e-2,                # 초기 learning rate
        lrf=3e-4,                # 최종 lr (cosine decay 끝값 비율)
        momentum=0.937,          # SGD/AdamW 모멘텀
        weight_decay=0.0005,     # L2 정규화 강도
        warmup_epochs=3.0,       # 워밍업 epoch 수
        patience=20,             # early stopping patience
        augment=True,            # 데이터 증강 사용 여부
        project="runs/train",    # 결과 저장 폴더
        name="yolov8n_custom",   # 실험 이름
        exist_ok=False,          # 폴더 중복 시 덮어쓰기 여부
        pretrained=False,         # 사전학습 가중치 사용 여부
    )

