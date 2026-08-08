#!/usr/bin/env python
"""
WalkerWalk에 도메인 랜덤화(DR)를 연결하는 패치.

repo 루트(= custom_envs/ 가 있는 폴더)에서 실행:
    python patch_walker_dr.py

배경:
  walker.py 에는 domain_randomize / domain_randomize_eval 함수가 이미 완성돼 있는데,
  custom_envs/dm_control_suite/__init__.py 의 "등록부"(_randomizer 딕셔너리)에
  WalkerWalk 만 빠져 있어서 DR 이 실제로 적용되지 않는 상태였습니다.
  이 스크립트는 그 딕셔너리에 WalkerWalk 항목 2줄을 추가할 뿐입니다.

이미 적용돼 있으면 건너뜁니다 (여러 번 실행해도 안전).
"""
import os
import sys

TARGET = os.path.join("custom_envs", "dm_control_suite", "__init__.py")
if not os.path.isfile(TARGET):
    sys.exit("ERROR: repo 루트(custom_envs/ 가 보이는 곳)에서 실행하세요.")

s = open(TARGET).read()

edits = [
    # (라벨, 기준 문자열, 바꿀 문자열)
    ("학습용 randomizer 등록",
     '  "CheetahRun" : cheetah.domain_randomize,\n}',
     '  "CheetahRun" : cheetah.domain_randomize,\n'
     '  "WalkerWalk" : walker.domain_randomize,\n}'),
    ("평가용 randomizer 등록",
     '  "CheetahRun" : cheetah.domain_randomize_eval,\n}',
     '  "CheetahRun" : cheetah.domain_randomize_eval,\n'
     '  "WalkerWalk" : walker.domain_randomize_eval,\n}'),
]

for label, old, new in edits:
    if new in s:
        print(f"[skip] {label}: 이미 적용됨")
        continue
    if old not in s:
        sys.exit(f"[ERROR] {label}: 기준 문자열을 못 찾음")
    s = s.replace(old, new, 1)
    print(f"[ok]   {label}: 적용")

open(TARGET, "w").write(s)
print("\n완료: WalkerWalk 에 도메인 랜덤화가 연결되었습니다.")
