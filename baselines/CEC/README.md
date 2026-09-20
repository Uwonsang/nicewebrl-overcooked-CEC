# CEC Baselines

Training and evaluation code for **Cross-Environment Coordination (CEC)** and comparison baselines on Overcooked (JAX/JaxMARL).

## Files

### Training

| File | Description |
|------|-------------|
| [ippo_general.py](ippo_general.py) | IPPO trained across 5 procedurally generated 9x9 Overcooked layouts (CEC training regime) |
| [ippo_general_population.py](ippo_general_population.py) | IPPO population training — trains a pool of agents for FCP partner generation |
| [fcp_general.py](fcp_general.py) | Fictitious Co-Play (FCP) — loads a population and trains a coordinator agent |
| [e3t.py](e3t.py) | E3T — trains with entropy-regularised teammate randomization across the 5 layouts |

### Evaluation

| File | Description |
|------|-------------|
| [test_general.py](test_general.py) | Cross-play evaluation: loads a checkpoint and rolls out against all partner types across layouts |
| [test_general_pcg.py](test_general_pcg.py) | Same as above but on procedurally generated (held-out) layouts |
| [test_general_check.py](test_general_check.py) | Sanity-check variant of `test_general.py` |
| [test_e3t.py](test_e3t.py) | Evaluation script specific to E3T checkpoints (includes t-SNE embedding) |
| [test_oracle.py](test_oracle.py) | Evaluates an oracle agent (trained per-layout) as an upper-bound baseline |
| [test_cross_env.py](test_cross_env.py) | Evaluates cross-environment generalization |
| [test_all_models_cross.py](test_all_models_cross.py) | Batch evaluation across all model types (IPPO, E3T, FCP, CEC, CEC_POP_ART) |
| [cross_algo.py](cross_algo.py) | Shared cross-play rollout logic used by eval scripts |

### Networks

| File | Description |
|------|-------------|
| [actor_networks.py](actor_networks.py) | `ScannedRNN` (LSTM), `ActorCriticRNN`, `ActorCriticE3T` — shared across all algorithms |

### Config & Scripts

| Path | Description |
|------|-------------|
| [config/ippo_final.yaml](config/ippo_final.yaml) | Default Hydra config (hyperparams, env kwargs, W&B settings) |
| [shell/bash_test_general.sh](shell/bash_test_general.sh) | Runs `test_general.py` for all layouts × all models |
| [shell/bash_e3t.sh](shell/bash_e3t.sh) | Training script for E3T |
| [shell/bash_fcp.sh](shell/bash_fcp.sh) | Training script for FCP |
| [shell/bash_ippo_pop.sh](shell/bash_ippo_pop.sh) | Training script for IPPO population |
| [sweep/](sweep/) | W&B sweep configs for hyperparameter search |
| [figures/](figures/) | Scripts for generating paper figures |

## Layouts

All training uses five 9×9 Overcooked layouts:

- `cramped_room_9`
- `asymm_advantages_9`
- `coord_ring_9`
- `counter_circuit_9`
- `forced_coord_9`

Set via `ENV_KWARGS.layout` in config or as a CLI override.

## Usage

**Train CEC (IPPO across environments):**
```bash
python baselines/CEC/ippo_general.py
```

**Evaluate all models across all layouts:**
```bash
bash baselines/CEC/shell/bash_test_general.sh <gpu_id>
```

**Evaluate a single model/layout:**
```bash
python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 model_name=CEC
```

Config overrides follow Hydra syntax. Key config fields:

| Key | Default | Notes |
|-----|---------|-------|
| `ENV_KWARGS.random_reset` | `False` | `True` = CEC regime, `False` = single-task |
| `TRAIN_KWARGS.ckpt_id` | `0` | Checkpoint index to save/load |
| `WANDB_MODE` | `disabled` | Set to `online` to log to W&B |
| `TEST_KWARGS.self_play` | `False` | `True` evaluates full cross-play matrix |

## Figures

Figure 관련 코드는 [figures/](figures/)에 있다. 저장 경로는 스크립트가
자동으로 만들며, 대부분 `baselines/CEC/figures/results/` 아래에 PNG를 만든다.
아래 명령은 repository root에서 실행하는 것을 기준으로 한다.

### 준비

W&B 그래프에는 `wandb`, `pandas`, `matplotlib`이 필요하며 해당 run을 읽을 수
있도록 먼저 로그인해야 한다.

```bash
wandb login
```

W&B run URL이
`https://wandb.ai/<entity>/<project>/runs/<run-id>`라면 세 값을 각각
`--entity`, `--project`, `--run-id`에 넣는다.

### W&B 학습 진단 그래프

[figures/analysis/](figures/analysis/)의 스크립트는 한 W&B run에서 다섯 layout의
history를 가져와 rolling mean 선 그래프를 만든다.

공통 옵션은 다음과 같다.

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--entity` | `overcooked_ai` | W&B entity |
| `--project` | `crossenv_ued_gradient` | W&B project |
| `--run-id` | 스크립트 내부 기본값 | W&B run ID |
| `--smooth-window` | `50` | history row 단위 rolling-mean window; `1`이면 smoothing 없음 |
| `--x-axis` | 대부분 `env_step` | `env_step` 또는 `update_step` |

`update_step`은 run config의 `NUM_ENVS * NUM_STEPS`를 사용해 `env_step`에서
계산한다. 따라서 W&B run config에 두 값이 있어야 한다.

| 스크립트 | 그리는 값 | 추가 옵션 | 현재 `ippo_general_gradient.py` run과 호환 |
|---|---|---|---|
| `train_returns_graph.py` | layout별 `train_returns/<layout>` | 없음 | 예 |
| `eval_graph.py` | layout별 `eval/<layout>` | 없음 | 예 |
| `target_raw_graph.py` | raw target 평균 `target_raw/<layout>/mean` | 없음 | 예 |
| `target_popart_graph.py` | PopArt target 평균 `target_popart/<layout>/mean` | 없음 | PopArt run만 |
| `td_error_graph.py` | layout별 `td_error/<layout>/<metric>` | `--td-metric mean_abs|rmse` | `rmse`만 예 |
| `grad_norm_graph.py` | layout별 actor/value gradient norm | `--loss-type actor|value` | 아니오: 예전 `grad_conflict_*/norm/*` 키 사용 |
| `share_gradient_graph.py` | layout별 weighted gradient share 및 균등 점유율 대비 편차 | `--loss-type actor|value`, `--view deviation|raw` | 아니오: 제거된 `grad_share_weighted_*` 키 사용 |
| `absolute_contribution_graph.py` | `sample_share * gradient norm`의 layout별 선 또는 누적 면적 | `--loss-type actor|value`, `--view line|stack` | 아니오: gradient norm에 예전 키 사용 |

기본 실행 형태는 다음과 같다.

```bash
python baselines/CEC/figures/analysis/<script>.py \
  --entity overcooked_ai \
  --project <project> \
  --run-id <run-id> \
  --x-axis update_step \
  --smooth-window 20
```

현재 로깅과 바로 호환되는 예시는 다음과 같다.

```bash
python baselines/CEC/figures/analysis/train_returns_graph.py \
  --project crossenv_ued_aaai --run-id <run-id>

python baselines/CEC/figures/analysis/eval_graph.py \
  --project crossenv_ued_aaai --run-id <run-id> --x-axis update_step

python baselines/CEC/figures/analysis/target_raw_graph.py \
  --project crossenv_ued_aaai --run-id <run-id>

python baselines/CEC/figures/analysis/td_error_graph.py \
  --project crossenv_ued_aaai --run-id <run-id> --td-metric rmse

python baselines/CEC/figures/analysis/target_popart_graph.py \
  --project crossenv_ued_aaai --run-id <popart-run-id>
```

출력은 다음 위치에 저장된다.

```text
baselines/CEC/figures/results/<script_name>/<metric>_<model_name>_<run-id>_<x-axis>.png
```

파일명에 쓰이는 `model_name`은 W&B run config에서 읽으며, 없으면 `unknown`이
된다. 개별 옵션 전체는 다음처럼 확인할 수 있다.

```bash
python baselines/CEC/figures/analysis/td_error_graph.py --help
```

### `test_general` 결과 막대그래프

`test_general_graph.py`는 W&B가 아니라 `test_general.py`가 만든 CSV들을 읽어
알고리즘별 XP 성능을 비교한다. `bash_test_general.sh`와 동일하게 IPPO, E3T,
FCP, CEC-64, CEC-Prev, CEC-PopArt-64, CEC-PopArt-Prev 순서로 표시한다.

입력 디렉터리와 파일명은 다음 형식이어야 한다.

```text
baselines/CEC/results/test_general_<NUM_MODELS>/
  <ENV_NAME>/
    <ALGORITHM>_<map>_9_XP_results.csv
```

각 CSV에는 `reward`가 필요하다. `XP_ONLY=True`이면 `seed_1 != seed_2`인 행만
사용하므로 `seed_1`, `seed_2`도 필요하다.

```bash
python baselines/CEC/figures/test_general_graph.py

# bash_test_general.sh가 NUM_MODELS=3으로 만든 결과를 그림
python baselines/CEC/figures/test_general_graph.py NUM_MODELS=3 XP_ONLY=True
```

다음 네 파일을 만든다.

```text
baselines/CEC/figures/results/test_general_graph_<NUM_MODELS>/
  xp_per_map.png
  xp_overall.png
  xp_per_map_table.csv
  xp_overall_table.csv
```

`xp_per_map.png`의 error bar는 seed pair별 reward 평균의 SEM이고,
`xp_overall.png`의 error bar는 map별 평균 reward의 SEM이다.

### Cross-algorithm XP heatmap

`cross_algo_graph.py`는 `cross_algo.py`가 생성한 다음 CSV를 읽는다.

```text
baselines/CEC/results/cross_algo/<layout>_cross_algo_eval_onIK.csv
```

CSV에는 `algo_1`, `algo_2`, `reward` 열이 필요하다. 각 matrix는 자기 자신의
최댓값으로 나눠 0–1 범위로 정규화한다.

```bash
python baselines/CEC/figures/cross_algo_graph.py
```

다음 네 heatmap을 만든다.

```text
baselines/CEC/figures/results/cross_algo_graph/
  cross_algo_per_layout_directional.png
  cross_algo_overall_directional.png
  cross_algo_per_layout_symmetric.png
  cross_algo_overall_symmetric.png
```

Directional 버전은 행을 Agent 0, 열을 Agent 1로 유지한다. Symmetric 버전은
두 역할 배치 `(i, j)`와 `(j, i)`의 평균이다. Overall은 layout별 raw matrix를
먼저 평균한 뒤 정규화한다.

### Held-out layout 이미지 점검

`test_general_map_check.py`는 고정 seed로 생성한 held-out layout 100개를
렌더링한다. 학습 결과 그래프가 아니라 생성된 map 자체를 확인하기 위한
도구다.

```bash
python baselines/CEC/figures/test_general_map_check.py
```

출력은 `baselines/CEC/figures/map_check/held_out_layout_<index>.png`에 저장된다.

### W&B config 보정 도구

`update_wandb_config.py`는 figure가 아니라 출력 파일명에 사용하는
`model_name` 등의 run config를 보정하는 도구다. 기본은 dry-run이다.

```bash
# 변경 대상만 확인
python baselines/CEC/figures/update_wandb_config.py \
  --project crossenv_ued_aaai \
  --name-contains pop \
  --value CEC_POP

# 실제 적용
python baselines/CEC/figures/update_wandb_config.py \
  --project crossenv_ued_aaai \
  --name-contains pop \
  --value CEC_POP \
  --apply
```

기본 변경 key는 `model_name`이며 실행 중인 run은 건너뛴다. 정확한 run 이름은
`--display-name`, 다른 config key는 `--key`로 지정할 수 있다.
