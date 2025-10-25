#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=00:07:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=2:mem=1880gb
#PBS -j oe
#PBS -m abe
##PBS -l other=hyperthread

set -euo pipefail

module load cuda

# Set paths
export PYTHON="${HOME}/scratch/tanathep/py312/bin/python3"
export MODEL_PATH="${HOME}/scratch/model/DeepSeek-R1"
export DATA_PATH="${HOME}/scratch/ShareGPT_V3_unfiltered_cleaned_split.json"

# Set the master address and port
export DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE)
export MASTER_PORT=5000

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
-hostfile ${PBS_NODEFILE} \
-map-by ppr:1:node:PE=112 -oversubscribe -use-hwthread-cpus \
-bind-to none --report-bindings -display-map \
-tag-output -output-filename ${HOME}/run/sglang.${PBS_JOBID} \
-x PATH \
-x NCCL_DEBUG=INFO \
-x DIST_INIT_ADDR \
-x MASTER_PORT \
-x PYTHON \
-x MODEL_PATH \
-x DATA_PATH \
bash -c "time \${PYTHON} \
-m sglang.bench_offline_throughput \
--model-path \${MODEL_PATH} \
--dataset-path \${DATA_PATH} \
--num-prompts 2000 \
--load-format dummy \
--seed 2025 \
--dtype bfloat16 \
--tp 16 \
--nnodes 2 \
--node-rank \${OMPI_COMM_WORLD_RANK} \
--dist-init-addr \${DIST_INIT_ADDR}:\${MASTER_PORT} \
--enable-dp-attention \
--dp 2 \
--trust-remote-code \
--speculative-algorithm EAGLE \
--speculative-num-steps 1 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 2 \
--attention-backend fa3 \
--max-running-requests 32" \
2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}