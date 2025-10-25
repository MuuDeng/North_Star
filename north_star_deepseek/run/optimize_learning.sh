#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=420
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=2:mem=1880gb
#PBS -j oe
#PBS -M 393958790@qq.com
#PBS -m abe

module load cuda

# ===== NCCL/comm tuning =====
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ALGO=Tree            # ลอง Ring ด้วย แล้วเทียบ
export NCCL_PROTO=LL128          # ลอง Simple/LL/LL128 แล้วเทียบ
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=32
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_HCA=mlx5_0,mlx5_1 # ปรับตามคลัสเตอร์
export NCCL_IB_GID_INDEX=3       # RoCEv2 common; ปรับตามจริง
export NCCL_CROSS_NIC=1
export NCCL_NET_GDR_LEVEL=2      # เปิด GPUDirect RDMA (ถ้าระบบรองรับ)
export NCCL_DEBUG=INFO           # ชั่วคราวเพื่อวัดผล

# ===== คำนวณ addr =====
export DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE)

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile ${PBS_NODEFILE} \
  -map-by ppr:1:node:PE=112 -bind-to hwthread \
  --report-bindings -oversubscribe -use-hwthread-cpus \
  -display-map \
  -tag-output -output-filename ${HOME}/run/sglang.${PBS_JOBID} \
  -x PATH \
  -x NCCL_DEBUG -x NCCL_DEBUG_SUBSYS \
  -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_ALGO -x NCCL_PROTO \
  -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS \
  -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x NCCL_IB_GID_INDEX \
  -x NCCL_CROSS_NIC -x NCCL_NET_GDR_LEVEL \
  -x DIST_INIT_ADDR \
  bash -lc 'time /home/users/industry/ai-hpc/apacsc34/scratch/miniforge/bin/python3 \
  -m sglang.bench_offline_throughput \
  --model-path /home/users/industry/ai-hpc/apacsc34/scratch/model/DeepSeek-R1 \
  --dataset-path /home/users/industry/ai-hpc/apacsc34/scratch/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
  --tp 8 --pp 2 --nnodes 2 --trust-remote-code \
  --dist-init-addr ${DIST_INIT_ADDR}:5000 --node-rank ${OMPI_COMM_WORLD_NODE_RANK} \
  ' \
  2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
