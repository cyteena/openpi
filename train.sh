#!/usr/bin/env bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_dfm_libero --exp-name=default --overwrite