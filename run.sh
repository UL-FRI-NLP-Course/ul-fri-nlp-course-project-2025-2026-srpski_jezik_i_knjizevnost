#!/bin/bash
module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
unset PYTHONPATH
export PYTHONNOUSERSITE=1
/d/hpc/home/sa99594/envs/myenv/bin/python main.py "$@"