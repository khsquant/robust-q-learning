# Learning RL Agents

This directory is scoped to q-learning style agents for MuJoCo Playground
environments. The active training entrypoint is:

```bash
python learning/train.py policy=td3 task=CheetahRun
```

Supported policies are `sac`, `wdsac`, `td3`, `m2td3`, `gmmtd3`, and `wdtd3`.
Algorithm defaults live in `learning/configs`.
