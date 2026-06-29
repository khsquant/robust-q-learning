# Learning RL Agents

This directory is scoped to q-learning style agents for MuJoCo Playground
environments. The active training entrypoint is:

```bash
python learning/train.py policy=td3 task=CheetahRun
```

Supported policies are `sac`, `wdsac`, `td3`, `m2td3`, `bridgetd3`, `tc_bridgetd3`, `gmmtd3`, `tc_gmmtd3`, `wdtd3`, `rarl`, `vanilla_tc_m2td3`, `tc_rarl`, and `tc_m2td3`.
Algorithm defaults live in `learning/configs`.

Sweep launchers live in `commands/`; for example:

```bash
bash commands/bridgetd3_sweeps.sh 0
```
