# 오디오 기기 객체 탐지 과제

헤드셋, 무선 이어폰, 스피커를 대상으로 YOLOv8n을 fine-tuning했다.
로컬 RTX 5060에서 100 epochs 학습했으며, 테스트 mAP50은 0.8945이다.

## 제출 자료

- [보고서](YOLO_audio_report.docx)
- [학습 모델](best.pt)
- [5초 탐지 및 추적 영상](audio3_tracking_5s.mp4)
- [데이터 출처](DATA_SOURCES.md)

학습 과정과 성능표, 영상 결과는 보고서에 정리했다.

## 실행

Python 3.12와 CUDA용 PyTorch가 필요하다. 사용 버전은 torch 2.7.1+cu128, torchvision 0.22.1+cu128이다.

```bash
pip install -r requirements.txt
python track.py --source audio3.mp4 --model best.pt --output runs/demo
python train.py --data /absolute/path/to/data.yaml
```

원본 영상과 학습 데이터는 별도로 준비한다. `data.yaml`의 경로를 실제 데이터 폴더로 바꾸고, 클래스 순서를 headset, wireless_earbuds, speaker로 맞춘다.

영상은 신뢰도 기준 0.4로 조정한 시연 결과다. 헤드셋은 150프레임 중 2프레임에서 누락됐고, 테스트 헤드셋 표본은 5장뿐이다.

데이터 출처와 라이선스는 [DATA_SOURCES.md](DATA_SOURCES.md), Ultralytics 사용 조건은 [공식 라이선스](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)를 참고한다.
