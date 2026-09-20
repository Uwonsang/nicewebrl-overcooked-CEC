# CEC-UED training and diagnostic metrics

이 문서는 `ippo_general_gradient.py`가 Weights & Biases(W&B)에 기록하는
지표를 기준으로 정리한다. PopArt에만 해당하는 차이는 별도로 표시한다. 별도
언급이 없다면 actor와 critic은 shared trunk를 포함한다.

## 기록 시점

```text
LOG_INTERVAL = max(1, NUM_UPDATES // 100)
```

W&B의 global step은 0부터 시작하는 PPO `update_step`이다. 현재 코드의
`env_step`은 다음과 같이 기록된다.

$$
\text{env\_step}
= \text{update\_step}\times\text{NUM\_ENVS}\times\text{NUM\_STEPS}.
$$

지표는 기록 방식에 따라 세 종류로 나뉜다.

| 종류 | 기록 방식 |
|---|---|
| PPO loss, entropy, ratio, optimizer gradient norm, parameter weight norm, `returns` | interval 동안 finite 값의 평균 |
| target, critic RMSE, TD-error RMSE | 해당 logging update의 rollout snapshot |
| layout gradient, representation feature/rank | 해당 logging update에서만 계산한 pre-update snapshot |
| evaluation | 가장 최근 diagnostic evaluation 결과 |
| layout별 training return | interval에서 종료된 episode들의 평균 |

마지막 PPO update는 interval 경계와 일치하지 않더라도 기록된다. Layout
gradient와 representation feature/rank는 PPO parameter update 직전의 parameter
및 rollout로 계산된다. Weight norm은 각 PPO minibatch optimizer step 직전
parameter에서 계산하지만 W&B 값은 interval 동안 평균한다.

## 기본 학습 지표

| W&B key | 의미 |
|---|---|
| `update_step` | 0부터 시작하는 현재 PPO update index |
| `env_step` | `update_step * NUM_ENVS * NUM_STEPS` |
| `returns` | rollout의 완료 episode return 평균 |
| `total_loss` | actor loss + `VF_COEF` × value loss − `ENT_COEF` × entropy |
| `value_loss` | clipped PPO value loss |
| `actor_loss` | clipped PPO surrogate actor loss |
| `entropy` | policy entropy |
| `ratio` | 새 policy와 rollout policy의 probability ratio 평균 |
| `ratio_0` | optimizer epoch/minibatch 평균 전 `ratio`의 첫 원소 |
| `approx_kl` | `(ratio - 1) - logratio`의 평균 |
| `clip_frac` | `abs(ratio - 1) > CLIP_EPS`인 sample 비율 |

`returns`와 loss 계열은 logging interval 동안 finite 값만 누적해 평균한다.
`ratio_0`은 전체 ratio 통계가 아니라, update 결과를 정리하는 과정에서 남긴
첫 원소이므로 일반적인 모니터링에는 `ratio`가 더 적합하다.

## 표기법

- $N$: environment sample 수. 기본 설정에서는 `NUM_ENVS=256`.
- $T$: `NUM_STEPS`.
- $A$: `NUM_ACTORS`. 기본 Overcooked 설정에서는 environment당 두 actor이므로
  $A=2N$이다.
- $A_m=A/\texttt{NUM\_MINIBATCHES}$: 한 PPO minibatch의 actor 수.
- $g_i$: environment sample $i$ 또는 layout $i$의 gradient.
- $\theta_k$: parameter tree의 $k$번째 array leaf.
- $n_k$: $\theta_k$의 원소 수.
- $\Phi\in\mathbb{R}^{M\times D}$: rollout에서 수집한 penultimate feature
  matrix. 시간과 actor 축을 $M$개의 sample 축으로 합친다.
- $\sigma_1\ge\cdots\ge\sigma_r$: $\Phi$의 singular values.
- $\epsilon=10^{-8}$: 수치 안정성을 위한 작은 상수.
- `<layout>`: `cramped_room_9`, `asymm_advantages_9`, `coord_ring_9`,
  `counter_circuit_9`, `forced_coord_9` 중 하나.

## 실제 PPO gradient norm

이 지표들은 gradient clipping과 Adam 적용 전의 total PPO gradient를 각
minibatch에서 측정한 뒤 epoch/minibatch 및 logging interval에 걸쳐 평균한다.
Bias와 1D parameter를 포함한 모든 array leaf가 참여한다.

Global L2 norm은 다음과 같다.

$$
\|g\|_2=\sqrt{\sum_k\sum_j g_{k,j}^2}.
$$

현재 코드의 leaf-count weighted RMS는 다음과 같다.

$$
\operatorname{WeightedRMS}(g)=
\sqrt{
\frac{\sum_k n_k\|g_k\|_2^2}
{\sum_k n_k}
}.
$$

이는 일반적인 elementwise RMS
$\sqrt{\sum_k\|g_k\|_2^2/\sum_k n_k}$와 다르며, 큰 leaf에 더 큰 가중치를
준다.

| W&B key | 포함 범위 |
|---|---|
| `gradient_norm/global_norm` | 전체 parameter tree |
| `gradient_norm/weighted_rms_norm` | 전체 parameter tree |
| `gradient_norm/actor_weighted_rms_norm` | shared trunk와 actor branch/output |
| `gradient_norm/critic_weighted_rms_norm` | shared trunk와 critic branch/output |
| `gradient_norm/shared_weighted_rms_norm` | shared trunk |

Actor와 critic 값에는 같은 shared trunk가 각각 포함된다. 또한 total PPO
gradient에서 module만 선택하므로 순수 actor-loss gradient와 순수 value-loss
gradient를 분리한 값은 아니다.

## Target, critic 및 TD error

GAE의 TD residual과 advantage는 다음과 같다.

$$
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),
$$

$$
\hat A_t=\delta_t+
\gamma\lambda(1-d_t)\hat A_{t+1},
\qquad
y_t=\hat A_t+V(s_t).
$$

일반 CEC에서는 모두 raw reward scale이다.

| W&B key | 수식 및 의미 |
|---|---|
| `target_raw/mean` | $\operatorname{mean}(y_t)$ |
| `target_raw/<layout>/mean` | 해당 layout의 $\operatorname{mean}(y_t)$ |
| `critic/rmse` | $\sqrt{\operatorname{mean}((y_t-V(s_t))^2)}$ |
| `critic/<layout>/rmse` | 해당 layout의 critic RMSE |
| `td_error/rmse` | $\sqrt{\operatorname{mean}(\delta_t^2)}$ |
| `td_error/<layout>/rmse` | 해당 layout의 TD-error RMSE |
| `td_error/zero_reward_rmse` | `abs(reward) < 1e-8`인 transition의 TD-error RMSE |
| `td_error/nonzero_reward_rmse` | 나머지 transition의 TD-error RMSE |

### PopArt scale

PopArt는 raw target $y_t$를 다음과 같이 정규화한다.

$$
\tilde y_t=\frac{y_t-\mu}{\sigma}.
$$

| W&B key | scale |
|---|---|
| `target_raw/*` | 역정규화된 실제 reward scale |
| `target_popart/mean` | normalized target scale |
| `target_popart/<layout>/mean` | normalized target scale |
| `target_popart/<layout>/std` | normalized target scale |
| `critic/*/rmse` | normalized target/value scale |
| `td_error/*/rmse` | raw reward scale |

따라서 일반 CEC와 PopArt CEC의 `critic/rmse`는 같은 이름이어도 scale이
다르므로 직접적인 수치 비교에 주의해야 한다.

## Layout-family gradient

Layout gradient 진단에는 rollout 전체가 아니라 앞쪽
`GRAD_CONFLICT_WINDOW_STEPS` 구간을 사용한다.

Layout $l$에서 관측된 sample 수를 $n_l$, 전체 sample 수를
$n=\sum_l n_l$, layout 평균 gradient를 $g_l$이라 한다. 두 agent는 같은
environment layout에 속하며 gradient 계산에는 두 agent가 모두 포함된다.

### Sample share와 norm

$$
p_l=\frac{n_l}{n},
\qquad
G=\sum_l n_lg_l.
$$

| W&B key | 수식 |
|---|---|
| `sample_share/<layout>` | $p_l$ |
| `grad_norm_actor/<layout>` | $\|g_l^{\mathrm{actor}}\|_2$ |
| `grad_norm_critic/<layout>` | $\|g_l^{\mathrm{value}}\|_2$ |
| `grad_norm_actor/total` | $\|G^{\mathrm{actor}}\|_2/n$ |
| `grad_norm_critic/total` | $\|G^{\mathrm{value}}\|_2/n$ |

### Layout signed contribution

전체 combined gradient 방향으로 투영한 sample-weighted signed contribution은
다음과 같다.

$$
C_l^{\mathrm{signed}}=
p_l\|g_l\|_2\cos(g_l,G).
$$

| W&B key | 의미 |
|---|---|
| `grad_contribution_signed_actor/<layout>` | actor의 $C_l^{\mathrm{signed}}$ |
| `grad_contribution_signed_critic/<layout>` | value의 $C_l^{\mathrm{signed}}$ |

`signed`가 음수이면 해당 layout gradient가 전체 combined gradient 방향을
방해한다. Layout이 해당 rollout에 존재하지 않으면 관련 layout metric은
`NaN`이다. 현재 `gradient_conflict_utils.py`는 layout 간 cosine conflict와
magnitude contribution을 계산하거나 기록하지 않는다.

### Family gradient norm equalization

`ippo_general_gradient_constraint.py`는 PPO minibatch에서 각 layout family의
total-loss gradient를 별도로 계산한다. Minibatch에 존재하는 family들의 raw
gradient norm 평균을 목표 norm으로 사용한다.

$$
\bar n=\frac{1}{|\mathcal F|}\sum_{f\in\mathcal F}\|g_f\|_2,
\qquad
\tilde g_f=g_f\frac{\bar n}{\|g_f\|_2+\epsilon},
\qquad
g_{\mathrm{update}}=\frac{1}{|\mathcal F|}\sum_{f\in\mathcal F}\tilde g_f.
$$

방향은 유지하고 family별 gradient 크기만 동일하게 맞춘다. Minibatch에 없는
family는 결합에서 제외한다.

```text
family_gradient_norm/target
family_gradient_norm/raw/<layout>
family_gradient_norm/equalized/<layout>
```

## Representation weight metrics

현재 weight metric은 kernel뿐 아니라 bias와 learned 1D scale/shift를 포함한
모든 parameter leaf를 사용한다. SimBaV2의 update-level parameter norm과
맞추기 위해 각 PPO minibatch optimizer step 직전 parameter에서 계산한다.
먼저 한 PPO update의 epoch/minibatch 전체를 평균하고, W&B에는 다시 logging
interval 동안의 update 평균을 기록한다. 따라서 이 지표는 diagnostic snapshot이
아니다.

$$
\|\theta\|_2=
\sqrt{\sum_k\sum_j\theta_{k,j}^2}.
$$

$$
\operatorname{WeightedRMS}(\theta)=
\sqrt{
\frac{\sum_k n_k\|\theta_k\|_2^2}
{\sum_k n_k}
}.
$$

| W&B key | 포함 범위 |
|---|---|
| `representation_weight/weight_norm` | 전체 parameter tree의 global L2 norm |
| `representation_weight/actor_weight_norm` | shared와 actor branch/output의 global L2 norm |
| `representation_weight/critic_weight_norm` | shared와 critic branch/output의 global L2 norm |
| `representation_weight/shared_weight_norm` | shared encoder/RNN parameter의 global L2 norm |
| `representation_weight/weighted_rms_norm` | 전체 parameter tree의 weighted RMS |
| `representation_weight/actor_weighted_rms_norm` | shared와 actor branch/output |
| `representation_weight/critic_weighted_rms_norm` | shared와 critic branch/output |
| `representation_weight/shared_weighted_rms_norm` | shared encoder/RNN parameter의 weighted RMS |

PopArt는 output-preserving rescaling을 critic output layer에 적용하므로 일반
CEC와 PopArt CEC의 `critic_weight_norm`을 직접 비교할 때 주의해야 한다.

## Gradient kurtosis

각 PPO minibatch의 gradient clipping 및 Adam 적용 전 raw gradient 원소
$G_i$를 다음과 같이 변환한다.

$$
L_i=\log(|G_i|+\epsilon),\qquad \epsilon=10^{-8}.
$$

기록하는 값은 excess kurtosis가 아닌 Pearson kurtosis다.

$$
K=
\frac{\mathbb{E}[(L_i-\mu_L)^4]}
{\left(\mathbb{E}[(L_i-\mu_L)^2]\right)^2+10^{-12}}.
$$

```text
gradient_kurtosis/global
gradient_kurtosis/actor
gradient_kurtosis/critic
gradient_kurtosis/shared
```

Actor와 critic은 각각 shared trunk와 해당 branch/output을 포함한다. Shared는
`Conv_0`, `Conv_1`, `Dense_0`, `Dense_1`, `ScannedRNN_0`만 포함한다.
값이 클수록 log-absolute gradient 분포의 tail이 무겁거나 극단적인 gradient
원소가 존재한다는 의미다. 다른 optimizer-update 지표와 마찬가지로 interval
동안 수행된 PPO minibatch들의 평균을 W&B에 기록한다.

## Penultimate feature metrics

Shared recurrent trunk와 actor/critic의 마지막 output layer 직전 activation으로
$\Phi$를 각각 구성한다.
Feature를 mean-center하지 않으므로 아래 spectrum은 centered covariance가 아니라
uncentered feature matrix의 spectrum이다. SimBaV2의 별도 metric replay batch에
대응하여, CEC에서는 diagnostic logging update에서 첫 PPO epoch의 첫 minibatch가
사용할 actor permutation을 동일한 RNG로 재현한다. 해당 minibatch의 전체
`NUM_STEPS` trajectory와 actor별 초기 LSTM state를 사용하므로 episode reset과
recurrent history가 실제 PPO update와 일치한다. Representation은 첫 minibatch
update 직전 parameter에서 별도 diagnostic forward로 한 번만 계산하며 학습
gradient에는 포함되지 않는다.

같은 environment의 두 actor를 평균하거나 concatenate하지 않고 각 actor-time
feature를 독립적인 matrix row로 사용한다. Feature matrix의 sample 수는

$$
M=TA_m
$$

이다. 기본값에서는 $M=400\times256=102{,}400$다. 이 값들은 interval 평균이
아니라 해당 첫 epoch·첫 minibatch의 pre-update snapshot이다.

### Feature norm

$$
\operatorname{feature\_norm}=
\frac{1}{M}\sum_{m=1}^{M}\|\phi_m\|_2.
$$

```text
representation_feature/shared_feature_norm
representation_feature/actor_feature_norm
representation_feature/critic_feature_norm
```

### Singular-value scale와 concentration

$$
\texttt{normalized\_sigma\_1}=\frac{\sigma_1(\Phi)}{\sqrt M},
\qquad
\texttt{sigma\_1\_ratio}=
\frac{\sigma_1}{\sum_j\sigma_j}.
$$

```text
representation_feature/shared_normalized_sigma_1
representation_feature/actor_normalized_sigma_1
representation_feature/critic_normalized_sigma_1
representation_feature/shared_sigma_1_ratio
representation_feature/actor_sigma_1_ratio
representation_feature/critic_sigma_1_ratio
```

`normalized_sigma_1`의 제곱 $\sigma_1^2/M$이
$\Phi^\top\Phi/M$의 최대 eigenvalue다. 현재 feature는 centering하지 않으므로
이는 covariance가 아니라 uncentered second-moment matrix의 eigenvalue다.

### Intermediate path-total feature norm

각 hidden layer $q$에서 sample별 feature L2 norm의 평균을 먼저 계산한다.

$$
F_q=\frac{1}{M}\sum_m\|h_m^{(q)}\|_2.
$$

Shared total은 shared path의 layer norm들을 단순 합산한다.

$$
F_{\mathrm{shared,total}}=
\sum_{q\in\mathrm{shared}}F_q.
$$

Actor와 critic total은 이 shared total과 해당 branch의 layer norm들을
단순 합산한다.

$$
F_{\mathrm{actor,total}}=
\sum_{q\in\mathrm{shared}}F_q+
\sum_{q\in\mathrm{actor}}F_q,
$$

$$
F_{\mathrm{critic,total}}=
\sum_{q\in\mathrm{shared}}F_q+
\sum_{q\in\mathrm{critic}}F_q.
$$

```text
representation_feature_total/shared_feature_norm
representation_feature_total/actor_feature_norm
representation_feature_total/critic_feature_norm
```

이 값은 하나의 penultimate feature norm이 아니라 여러 layer norm의 합이므로
network depth와 architecture가 다른 모델 사이의 직접 비교에는 적합하지 않다.

## Representation rank metrics

기본 cutoff는 $c=0.01$, threshold는 $1-c=0.99$다.

### Lyle feature rank

$$
\operatorname{feature\_rank}=
\sum_j\mathbf 1\left(\frac{\sigma_j}{\sqrt M}>c\right).
$$

```text
representation_rank/shared_feature_rank
representation_rank/actor_feature_rank
representation_rank/critic_feature_rank
```

이 rank는 absolute feature scale에 영향을 받는다.

### Roy--Vetterli effective rank

$$
p_j=\frac{\sigma_j}{\sum_k\sigma_k},
\qquad
\operatorname{effective\_rank}=
\exp\left(-\sum_jp_j\log p_j\right).
$$

```text
representation_rank/shared_effective_rank_vetterli
representation_rank/actor_effective_rank_vetterli
representation_rank/critic_effective_rank_vetterli
```

Spectrum이 여러 방향에 균등하게 퍼질수록 커지고 한두 방향에 집중될수록
작아진다. Uniform feature rescaling에는 불변이다.

### Kumar srank

$$
\operatorname{srank}_{c}=
\min\left\{k:
\frac{\sum_{j=1}^{k}\sigma_j}{\sum_j\sigma_j}
\ge 1-c
\right\}.
$$

```text
representation_rank/shared_srank_kumar
representation_rank/actor_srank_kumar
representation_rank/critic_srank_kumar
```

### PCA approximate rank

$$
\operatorname{rank}_{\mathrm{PCA},c}=
\min\left\{k:
\frac{\sum_{j=1}^{k}\sigma_j^2}{\sum_j\sigma_j^2}
\ge 1-c
\right\}.
$$

```text
representation_rank/shared_approximate_rank_pca
representation_rank/actor_approximate_rank_pca
representation_rank/critic_approximate_rank_pca
```

### Numerical matrix rank

$$
\tau=\max(M,D)\,\epsilon_{\mathrm{machine}}\,\sigma_1,
\qquad
\operatorname{matrix\_rank}=
\sum_j\mathbf 1(\sigma_j>\tau).
$$

```text
representation_rank/shared_matrix_rank
representation_rank/actor_matrix_rank
representation_rank/critic_matrix_rank
```

## Evaluation과 return

| W&B key | 의미 |
|---|---|
| `eval/mean` | 고정 evaluation layout 평균 return |
| `eval/<layout>` | layout별 evaluation return |
| `eval_xp/mean` | human-proxy cross-play 평균 return |
| `eval_xp/<layout>` | layout별 human-proxy cross-play return |
| `eval_critic/<layout>/value_mean` | self-play eval 중 critic value 평균 |
| `eval_critic/<layout>/target_mean` | raw eval reward의 bootstrapped discounted-return 평균 |
| `eval_critic/<layout>/value_rmse` | value와 bootstrapped discounted return 사이 RMSE |
| `eval_critic/<layout>/td_error_rmse` | raw eval reward 기준 one-step TD-error RMSE |
| `eval_xp_critic/<layout>/value_mean` | human-proxy XP 중 main policy critic value 평균 |
| `eval_xp_critic/<layout>/target_mean` | XP raw reward의 bootstrapped discounted-return 평균 |
| `eval_xp_critic/<layout>/value_rmse` | XP value와 bootstrapped discounted return 사이 RMSE |
| `eval_xp_critic/<layout>/td_error_rmse` | XP raw reward 기준 one-step TD-error RMSE |
| `train_returns/<layout>` | logging interval에 종료된 해당 layout episode return 평균 |

Eval rollout은 `EVAL_KWARGS.num_steps`의 마지막 transition에서 항상 실제
terminal state에 도달한다고 가정하므로 horizon 이후 value는 0으로 둔다.
Self-play에서는 두 actor의 value를 모두 사용하며, XP에서는 학습된 main
policy가 차지하는 seat의 value만 사용한다. XP의 squared error는 두 seat와
모든 human-proxy seed에 걸쳐 평균한 뒤 제곱근을 취한다.

이 target과 TD error는 shaped reward가 아닌 raw environment reward 기준이다.
따라서 shaped reward가 남아 있는 학습 초반에는 training critic metric과 직접
비교할 때 reward 정의의 차이에 주의해야 한다.

## 참고 문헌

- [Roy & Vetterli, *The Effective Rank: A Measure of Effective
  Dimensionality*](https://zenodo.org/records/40328) (2007): entropy-based
  effective rank.
- [Kumar et al., Implicit Under-Parameterization Inhibits Data-Efficient Deep
  Reinforcement Learning](https://arxiv.org/abs/2010.14498): threshold srank.
- [Yang et al., Harnessing Structures for Value-Based Planning and
  Reinforcement Learning](https://arxiv.org/abs/1909.12255): PCA approximate
  rank.
- [Lyle et al., Understanding and Preventing Capacity Loss in Reinforcement
  Learning](https://arxiv.org/abs/2204.09560): threshold feature rank.
- [Lyle et al., Understanding Plasticity in Neural
  Networks](https://arxiv.org/abs/2303.01486): per-transition gradient
  covariance/cosine analysis.
