#!/usr/bin/env python
"""
run.py에 gmmsac를 등록하는 패치 스크립트.

learning/ 폴더 안에서 실행하세요:
    cd learning
    python patch_run.py

하는 일은 딱 3가지 (아래 코드에서 그대로 볼 수 있음):
  1) 파일 맨 위 import 목록에 gmmsac 모듈 2줄 추가
  2) train_gmmsac 디스패처 함수 정의 추가
  3) TRAINERS 딕셔너리에 "gmmsac" 항목 등록

이미 패치돼 있으면 건너뜁니다 (여러 번 실행해도 안전).
"""
import os
import sys

RUN_PY = "run.py"
if not os.path.isfile(RUN_PY):
    sys.exit("ERROR: learning/ 폴더 안에서 실행하세요 (run.py가 안 보임).")

src = open(RUN_PY).read()

# ── 편집 1: import 추가 ────────────────────────────────────────────────
import_anchor = "from agents.sac import networks as sac_networks"
import_added = (
    import_anchor
    + "\nfrom agents.gmmsac import train as gmmsac"
    + "\nfrom agents.gmmsac import networks as gmmsac_networks"
)

# ── 편집 2: 디스패처 함수 추가 (train_td3 정의 바로 앞에 끼워넣음) ──────
dispatcher = '''def train_gmmsac(cfg, randomization_fn, env, eval_env=None):
    gmmsac_params = _sac_config(cfg.task)
    gmmsac_params.dr_augmented_critic = _cfg_flag(cfg, "dr_augmented_critic")
    _maybe_override_config(gmmsac_params, cfg)
    gmmsac_training_params = dict(gmmsac_params)
    wandb_name = f"{cfg.task}.{cfg.policy}.seed={cfg.seed}.beta={cfg.beta}"
    _init_wandb(cfg, wandb_name)

    if "network_factory" in gmmsac_params:
        del gmmsac_training_params["network_factory"]
        if not cfg.asymmetric_critic:
            gmmsac_params.network_factory.value_obs_key = "state"
        network_factory = functools.partial(
            gmmsac_networks.make_gmmsac_networks,
            **gmmsac_params.network_factory,
        )
    else:
        network_factory = gmmsac_networks.make_gmmsac_networks

    train_fn = functools.partial(
        gmmsac.train,
        **dict(gmmsac_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress_fn, use_wandb=cfg.use_wandb),
        randomization_fn=_adv_randomizer(cfg.task, randomization_fn),
        eval_randomization_fn=randomization_fn,
        dr_train_ratio=cfg.dr_train_ratio,
        seed=cfg.seed,
        beta=cfg.beta,
    )
    return train_fn(environment=env)


'''
td3_anchor = "def train_td3(cfg, randomization_fn, env, eval_env=None):"

# ── 편집 3: TRAINERS 딕셔너리에 등록 ──────────────────────────────────
trainers_anchor = '    "gmmtd3": train_gmmtd3,'
trainers_added = trainers_anchor + '\n    "gmmsac": train_gmmsac,'

edits = [
    ("import 2줄", import_anchor, import_added),
    ("train_gmmsac 디스패처", td3_anchor, dispatcher + td3_anchor),
    ("TRAINERS 등록", trainers_anchor, trainers_added),
]

for label, old, new in edits:
    if new in src:
        print(f"[skip] {label}: 이미 적용됨")
        continue
    if old not in src:
        sys.exit(f"[ERROR] {label}: 기준 문자열을 못 찾음 -> {old[:50]!r}")
    src = src.replace(old, new, 1)
    print(f"[ok]   {label}: 적용")

open(RUN_PY, "w").write(src)
print("\n완료: run.py에 gmmsac가 등록되었습니다.")
