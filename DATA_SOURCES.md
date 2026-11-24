# 데이터 출처와 처리

- HEADPHONES&EARBUDS v2, Fadwas Workspace: https://universe.roboflow.com/fadwas-workspace-0qt4y/headphones-earbuds/dataset/2 (제공자 표기 Public Domain)
- speaker v1, YOLO: https://universe.roboflow.com/yolo-cnyhz/speaker-qbqrv-pqco2/dataset/1 (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/)

이어폰과 헤드폰의 기존 클래스 번호를 1과 0으로 재매핑했다. 스피커 라벨은 2로 통일하고 다각형 라벨은 외접 사각형으로 바꿨다. 파일명을 해시로 변경했다. 스피커는 dHash 거리 6 이하 연결 그룹을 같은 분할에 두었다(seed 42). 정확한 픽셀 중복도 검사했다. 유사 장면의 완전한 분리는 보장하지 않는다.

헤드셋·이어폰은 원본 164장에 기존 증강본을 포함하여 394장, 스피커는 1102장이다. 총 1496장은 독립 원본 이미지 수를 뜻하지 않는다.

| 분할 | 이미지 | 헤드셋 객체 | 이어폰 객체 | 스피커 객체 |
|---|---:|---:|---:|---:|
| train |1116|183|159|771|
| val |254|16|17|221|
| test |126|5|11|110|

일부 이어폰 라벨은 낱개, 일부는 한 쌍 단위다. 일부 스피커 자료에 포장 사진과 추가 물체 라벨 누락이 있다. 전체 재라벨링은 하지 않았다. 데이터셋 원본은 이 저장소에 재배포하지 않는다.
