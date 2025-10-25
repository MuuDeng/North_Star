#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=420
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=2:mem=1880gb
#PBS -j oe
#PBS -m abe

module load cuda

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_ALGO=Tree
export NCCL_MIN_NRINGS=4
export DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE)

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
    -hostfile ${PBS_NODEFILE} \
    -map-by ppr:2:node:PE=56 -oversubscribe -use-hwthread-cpus \
    -bind-to none --report-bindings -display-map \
    -tag-output -output-filename ${HOME}/run/sglang.${PBS_JOBID} \
    -x PATH -x NCCL_DEBUG -x NCCL_ALGO -x NCCL_MIN_NRINGS -x DIST_INIT_ADDR \
bash -c 'time /home/users/industry/ai-hpc/apacsc34/scratch/nattanon/py312/bin/python3 \
    -m sglang.bench_offline_throughput \
    --model-path /home/users/industry/ai-hpc/apacsc34/scratch/model/DeepSeek-R1 \
    --dataset-path ${HOME}/scratch/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 2000 \
    --load-format dummy \
    --seed 2025 \
    --dtype bfloat16 \
    --tp 8 \
    --micro-batch-size 4 \
    --max-batch-size 32 \
    --sequence-length 4096 \
    --activation-checkpointing \
    --nnodes 2 \
    --trust-remote-code \
    --dist-init-addr ${DIST_INIT_ADDR}:5000 \
    --node-rank ${OMPI_COMM_WORLD_RANK}' \
2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
