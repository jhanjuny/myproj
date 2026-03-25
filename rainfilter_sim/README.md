
```

### 프로젝트 구조
```
rainfilter_sim/
├── src/
│   ├── __init__.py          # 패키지 초기화
│   ├── geometry.py          # 거름망 3D 형상 정의
│   ├── particles.py         # 입자 물리 엔진 (SPH 기반)
│   ├── simulation.py        # 시뮬레이션 메인 루프
│   └── visualization.py     # 3D 시각화 (PyVista/Open3D)
├── configs/
│   └── sim_params.json      # 시뮬레이션 파라미터
├── outputs/                 # 결과 영상/이미지/데이터
├── experiments/             # 다양한 조건 실험
├── docs/                    # 문서/노트
└── README.md