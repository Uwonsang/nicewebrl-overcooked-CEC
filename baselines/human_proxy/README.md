# Human Proxy (Behaviour Cloning)

Overcooked-AI human-human trial 데이터로 behaviour cloning을 학습해, CEC/IPPO와 동일한
9×9 observation 포맷을 쓰는 "human proxy" 에이전트를 만듭니다. 레이아웃별로 데이터와
모델을 분리해서 학습합니다.

## 파일 구성

| 파일 | 설명 |
|------|------|
| [layouts.py](layouts.py) | raw human 데이터 그리드를 jaxmarl의 고정 9×9 `Overcooked` 레이아웃 포맷으로 변환 |
| [preprocess.py](preprocess.py) | raw trial CSV를 레이아웃별 `(obs, action)` 쌍으로 변환. `Overcooked.get_obs`를 직접 호출해서 실제 환경과 observation이 완전히 동일하도록 함 |
| [bc_agent.py](bc_agent.py) | 사람 행동에 대해 cross-entropy로 학습하는 Flax CNN 분류기 (Hydra + W&B) |
| [config/bc.yaml](config/bc.yaml) | Hydra 설정 (하이퍼파라미터, 데이터 경로, 레이아웃, W&B 설정) |
| [data/origin/](data/origin/) | raw `2019_hh_trials.csv` |
| [data/processed/](data/processed/) | 전처리된 레이아웃별 `.npz` |

## 데이터 & 레이아웃 매핑

raw CSV는 Overcooked-AI 웹 실험의 원본(미가공) 로그입니다 — 한 row가 한 타임스텝이고,
전체 JSON `state`, `joint_action`, 그리고 그 레이아웃 고유의 그리드 문자열을 담고 있습니다.
`jaxmarl`의 `Overcooked` 환경은 9×9 보드를 하드코딩하고 있는데, raw 레이아웃은 그보다
작기 때문에(예: `cramped_room`은 4×5), `layouts.py`가 raw 그리드를 9×9 캔버스의
**왼쪽 상단**에 그대로 배치하고 나머지는 벽으로 채웁니다 (왼쪽 상단 정렬이라 좌표 오프셋이
필요 없습니다). 2019 데이터의 레이아웃 중 5개만 jaxmarl에 대응되는 레이아웃이 있습니다:

| raw `layout_name` | jaxmarl 레이아웃 |
|---|---|
| `cramped_room` | `cramped_room` |
| `coordination_ring` | `coord_ring` |
| `asymmetric_advantages` | `asymm_advantages` |
| `random0` | `forced_coord` |
| `random3` | `counter_circuit` |

**2020 데이터는 지원하지 않습니다.** 8개 레이아웃 전부 폭이 9칸을 넘거나(9×9에 안 들어감),
tomato를 실제 주문 재료로 사용하는데 `jaxmarl`의 `Overcooked`에는 tomato 오브젝트 타입 자체가
없습니다 (`common.py`의 `OBJECT_TO_INDEX`에 `tomato` 항목 없음). 쓰려면 이 파이프라인이
아니라 jaxmarl 환경 자체를 확장해야 합니다.

## 사용법

**raw CSV를 레이아웃별 train/test `.npz`로 전처리:**
```bash
python -m baselines.human_proxy.preprocess
```
`--csv`/`--out_dir` 기본값은 각각 `data/origin/2019_hh_trials.csv`, `data/processed`이며,
필요하면 오버라이드할 수 있습니다. 레이아웃마다 `{stem}_{jax_layout}_train.npz` /
`_test.npz` 형태로 따로 저장됩니다 (예: `2019_hh_trials_cramped_room_train.npz`).

**학습 (레이아웃 하나씩):**
```bash
python -m baselines.human_proxy.bc_agent LAYOUT=cramped_room
```
`LAYOUT`은 `cramped_room`, `coord_ring`, `asymm_advantages`, `forced_coord`,
`counter_circuit` 중 하나입니다. 다른 config 값도 Hydra 문법으로 오버라이드할 수 있습니다:
```bash
python -m baselines.human_proxy.bc_agent LAYOUT=counter_circuit NUM_EPOCHS=100 BATCH_SIZE=512 WANDB_MODE=online
```

| Key | 기본값 | 설명 |
|-----|---------|-------|
| `LAYOUT` | `cramped_room` | 학습할 레이아웃 (하나씩 학습) |
| `DATA_DIR` / `DATA_STEM` | `baselines/human_proxy/data/processed` / `2019_hh_trials` | 전처리된 데이터 위치 (실제로는 `{DATA_STEM}_{LAYOUT}_{split}.npz`를 로드) |
| `NUM_EPOCHS` | `50` | 전체 데이터셋 반복 횟수 |
| `BATCH_SIZE` | `256` | |
| `CKPT_DIR` | `baselines/human_proxy/checkpoints` | 학습된 파라미터(pickle) 저장 위치, `bc_overcooked_{LAYOUT}_seed{SEED}.pkl`로 저장 |
| `WANDB_MODE` | `disabled` | `online`으로 바꾸면 W&B에 로깅 |

같은 trial의 두 플레이어 (obs, action) 쌍을 모두 독립적인 학습 샘플로 사용합니다.
train/test는 `(trial_id, player_0_id, player_1_id)` 단위로 분리해서, 같은 사람 플레이
세션이 양쪽에 섞이지 않도록 했습니다.

**시드 여러 개 돌리기 (레이아웃 하나, seed 0~5):**
```bash
bash baselines/human_proxy/shell/bash_bc_sweep.sh <gpu_id> cramped_room
```
`SEED`별로 W&B run과 체크포인트(`bc_overcooked_{LAYOUT}_seed{SEED}.pkl`)가 따로 저장되므로,
W&B에서 같은 레이아웃의 seed들을 겹쳐서 비교할 수 있습니다. 5개 레이아웃 전부 돌리려면
레이아웃을 바꿔가며 5번 실행하면 됩니다 (`bash_ippo_pop.sh`와 동일한 패턴).

## 트러블슈팅: cuSolver 에러

GPU에서 `CustomCall failed: ... gpusolverDnCreate(&handle) failed: cuSolver internal error`가
뜨면, 네트워크의 `orthogonal()` 초기화(QR 분해, cuSolver 사용)가 JAX의 기본 GPU 메모리
preallocation과 충돌해서 그렇습니다. 아래처럼 preallocation을 꺼주면 해결됩니다
(`bash_bc_sweep.sh`에는 이미 반영되어 있습니다):
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m baselines.human_proxy.bc_agent LAYOUT=cramped_room
```

## 참고: 학습 시 환경(env)을 쓰지 않습니다

`bc_agent.py`는 학습 시점에 jaxmarl 환경을 직접 실행하지 않습니다. 환경은 `preprocess.py`
단계에서 `get_obs()`를 한 번 호출해 observation을 미리 만들어두는 데만 쓰이고,
`bc_agent.py`는 그 결과 `.npz`를 불러와 순수 지도학습(supervised learning)으로
cross-entropy loss를 최소화할 뿐입니다.
