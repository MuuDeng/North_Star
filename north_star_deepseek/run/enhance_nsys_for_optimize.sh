#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -m abe

# ========= 1) Base env =========
module load cuda

MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=$((20000 + RANDOM % 10000))   # unique port per run
export MASTER_ADDR MASTER_PORT

# (ถ้ารู้ว่าเป็น H100/Blackwell ตั้ง arch เพื่อลดเวลา compile extension)
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO

# ========= 2) Run (8 ranks/node × 2 nodes = 16 ranks) =========
time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile "${PBS_NODEFILE}" \
  -map-by ppr:8:node:PE=14 -bind-to core --report-bindings \
  -tag-output -output-filename "${HOME}/scratch/tanathep/run/sglang.${PBS_JOBID}" \
  -x PATH -x MASTER_ADDR -x MASTER_PORT -x NCCL_DEBUG \
  -x CUDA_VISIBLE_DEVICES -x CUDA_DEVICE_ORDER -x TORCH_CUDA_ARCH_LIST \
  bash -lc '
    # ===== paths (กำหนดใน subshell นี้เพื่อกัน env หาย) =====
    NSYS_BIN=/home/users/industry/ai-hpc/apacsc34/nsight-systems-2025.5.1/bin/nsys
    PYTHON=${HOME}/scratch/tanathep/py312/bin/python3
    MODEL=${HOME}/scratch/model/DeepSeek-R1
    DATA=${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
    OUTDIR=${HOME}/scratch/tanathep/profiling/nsys_reports
    mkdir -p "$OUTDIR"

    # ===== rank info =====
    RANK=${OMPI_COMM_WORLD_RANK}
    LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
    NODE_LOCAL_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE:-8}
    NODE_RANK=$(( RANK / NODE_LOCAL_SIZE ))
    WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}

    # ===== auto-detect network interface ที่ไปถึง MASTER_ADDR =====
    DETECTED_IF=$(ip -o route get "$MASTER_ADDR" 2>/dev/null | awk '"'"'{for(i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}'"'"')
    if [ -z "$DETECTED_IF" ]; then
      # fallback: ใช้ default route
      DETECTED_IF=$(ip route show default 2>/dev/null | awk '"'"'{for(i=1;i<=NF;i++) if ($i=="dev") {print $(i+1); exit}}'"'"')
    fi
    export GLOO_SOCKET_IFNAME="$DETECTED_IF"
    export NCCL_SOCKET_IFNAME="$DETECTED_IF"

    echo "[DBG] host=$(hostname) if=$DETECTED_IF rank=$RANK local=$LOCAL_RANK node=$NODE_RANK world=$WORLD_SIZE init=${MASTER_ADDR}:${MASTER_PORT}" >&2

    # ===== run: profile เฉพาะ local_rank=0 ต่อโหนด =====
    if [ "$LOCAL_RANK" -eq 0 ]; then
      echo "[INFO] Profiling rank $RANK (node $NODE_RANK, local $LOCAL_RANK)"
      "$NSYS_BIN" profile \
        --output="$OUTDIR/sglang_profile.${PBS_JOBID}.node${NODE_RANK}.rank${RANK}" \
        --trace=cuda,nvtx,mpi,osrt \
        --capture-range=nvtx --capture-range-end=stop \
        --sample=none --force-overwrite=true \
        "$PYTHON" -m sglang.bench_offline_throughput \
          --model-path "$MODEL" \
          --dataset-path "$DATA" \
          --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
          --tp 16 --nnodes 2 --trust-remote-code \
          --dist-init-addr ${MASTER_ADDR}:${MASTER_PORT} \
          --node-rank ${NODE_RANK}
    else
      echo "[INFO] Rank $RANK (node $NODE_RANK, local $LOCAL_RANK) running baseline."
      "$PYTHON" -m sglang.bench_offline_throughput \
        --model-path "$MODEL" \
        --dataset-path "$DATA" \
        --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
        --tp 16 --nnodes 2 --trust-remote-code \
        --dist-init-addr ${MASTER_ADDR}:${MASTER_PORT} \
        --node-rank ${NODE_RANK}
    fi
  ' 2>&1 | tee "${HOME}/run/stdout.sglang.${PBS_JOBID}"
