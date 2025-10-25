#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -m abe

set -euo pipefail
set -x

module load cuda

export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=5000
export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=ib0
# export GLOO_SOCKET_IFNAME=ib0

# พรีคอมไพล์ DeepGEMM (เร็วขึ้นรอบจริง)
/usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun -hostfile "$PBS_NODEFILE" -map-by ppr:1:node -np 2 \
  bash -lc "/home/users/industry/ai-hpc/apacsc34/scratch/miniforge/bin/python3 -m sglang.compile_deep_gemm"

PROFILE=${PROFILE:-0}   # set PROFILE=1 เมื่ออยากเก็บ Nsight

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile "$PBS_NODEFILE" \
  -map-by ppr:8:node:PE=14 -bind-to core --report-bindings \
  -tag-output -output-filename "${HOME}/run/sglang.${PBS_JOBID}" \
  -x PATH -x LD_LIBRARY_PATH \
  -x MASTER_ADDR -x MASTER_PORT \
  -x NCCL_DEBUG -x NCCL_SOCKET_IFNAME -x GLOO_SOCKET_IFNAME \
  bash -lc '
    set -euo pipefail
    set -x

    # OMPI envs (fallback ถ้าไม่มี ให้แจ้งใน log)
    export RANK=${OMPI_COMM_WORLD_RANK:-0}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-1}
    export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    export OMPI_LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE:-8}

    # ค้นหา GPU ต่อโหนดจริง (ถ้าไม่มี nvidia-smi จะจับเป็น 0)
    if command -v nvidia-smi >/dev/null 2>&1; then
      GPUS_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
      GPUS_PER_NODE=0
    fi

    if [ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ]; then
      echo "WARNING: no GPUs detected on this node (GPUS_PER_NODE=${GPUS_PER_NODE})."
      echo "Proceeding without setting CUDA_VISIBLE_DEVICES."
      export CUDA_VISIBLE_DEVICES=
    else
      # assign GPU id robustly: use modulo in case LOCAL_RANK >= GPUS_PER_NODE
      GPU_ID=$(( LOCAL_RANK % GPUS_PER_NODE ))
      export CUDA_VISIBLE_DEVICES=$GPU_ID
      echo "Detected GPUS_PER_NODE=${GPUS_PER_NODE}. LOCAL_RANK=${LOCAL_RANK} => CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi

    export NODE_RANK=$(( RANK / ${OMPI_LOCAL_SIZE} ))
    export OMP_NUM_THREADS=14

    # debug prints
    echo "=== debug env ==="
    echo "RANK=$RANK"
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "LOCAL_RANK=$LOCAL_RANK"
    echo "OMPI_LOCAL_SIZE=${OMPI_LOCAL_SIZE}"
    echo "NODE_RANK=$NODE_RANK"
    echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
    echo "MASTER_ADDR=${MASTER_ADDR}"
    echo "MASTER_PORT=${MASTER_PORT}"
    echo "================="

    # โปรไฟล์เฉพาะ local rank 0 ต่อโหนด + NVTX capture (ไฟล์เล็กกว่าและสัญญาณชัด)
    if [ "$PROFILE" = "1" ] && [ "$LOCAL_RANK" -eq 0 ]; then
      mkdir -p ${HOME}/profiling/nsys_reports
      NSYS="/home/users/industry/ai-hpc/apacsc34/nsight-systems-2025.5.1/bin/nsys profile \
            --output=${HOME}/profiling/nsys_reports/sglang.${PBS_JOBID}.node${NODE_RANK} \
            --trace=cuda,nvtx,mpi,osrt \
            --capture-range=nvtx --capture-range-end=stop \
            --sample=none --force-overwrite=true"
    else
      NSYS=""
    fi

    # รันจริง (ใส่ NVTX ครอบช่วง timed ในโค้ด ถ้าจะใช้ capture-range)
    ${NSYS:+$NSYS } /home/users/industry/ai-hpc/apacsc34/scratch/miniforge/bin/python3 -m sglang.bench_offline_throughput \
      --model-path ${HOME}/scratch/model/DeepSeek-R1 \
      --dataset-path ${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
      --tp 16 --nnodes 2 --trust-remote-code \
      --dist-init-addr ${MASTER_ADDR}:${MASTER_PORT} \
      --node-rank ${NODE_RANK}
  '
