#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=420
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -m abe

module load cuda

# คำนวณที่อยู่ init ของ process 0 บนโหนดแรก
export DIST_INIT_ADDR=$(getent hosts $(head -n1 $PBS_NODEFILE) | awk '{print $1}')

# ค่าที่แนะนำสำหรับ NCCL/UCX (ปรับตามคลัสเตอร์ได้)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_NET_GDR_LEVEL=PHB

# ลดการแย่ง CPU ถ้าแอปไม่ได้ใช้ OpenMP หนัก ๆ
export OMP_NUM_THREADS=1

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile ${PBS_NODEFILE} \
  -map-by ppr:8:node:pe=14 -bind-to core --rank-by socket \
  -mca pml ucx -mca osc ucx \
  --report-bindings -display-map -tag-output \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING \
  -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA \
  -x NCCL_IB_GID_INDEX -x NCCL_IB_QPS_PER_CONNECTION \
  -x NCCL_NET_GDR_LEVEL \
  -x DIST_INIT_ADDR \
  bash -lc '
    export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
    outdir=${HOME}/scratch/nattanon/profiling/nsys_reports
    mkdir -p "$outdir"
    export PATH="/home/users/industry/ai-hpc/apacsc34/nsight-systems-2025.5.1/bin:$PATH"

    # ----- Nsight Systems: จับทั้งรัน (ง่ายสุด) -----
    nsys profile \
      --force-overwrite=true \
      --gpu-metrics-devices=cuda-visible \
      -t cuda,nvtx \
      -o ${outdir}/sglang_${PBS_JOBID}_rank${OMPI_COMM_WORLD_RANK} \
      ${HOME}/scratch/tanathep/py312/bin/python3 -m sglang.bench_offline_throughput \
        --model-path ${HOME}/scratch/model/DeepSeek-R1 \
        --dataset-path ${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
        --tp 8 --nnodes 2 --trust-remote-code \
        --dist-init-addr ${DIST_INIT_ADDR}:5000 \
        --node-rank ${OMPI_COMM_WORLD_NODE_RANK}

    # ----- Export เป็น SQLite: ชื่อไฟล์ต้องตรงกับ -o ข้างบน -----
    nsys export --sqlite true \
      -o ${outdir}/sglang_${PBS_JOBID}_rank${OMPI_COMM_WORLD_RANK}.sqlite \
      ${outdir}/sglang_${PBS_JOBID}_rank${OMPI_COMM_WORLD_RANK}.nsys-rep
  ' \
2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
