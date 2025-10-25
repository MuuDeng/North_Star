#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=420
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -m abe
module load cuda

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile ${PBS_NODEFILE} \
  -map-by ppr:8:node:pe=14 -bind-to core \
  -mca pml ucx -mca osc ucx \
  --report-bindings -display-map -tag-output \
  -x NCCL_DEBUG=INFO -x NCCL_ASYNC_ERROR_HANDLING=1 \
  -x NCCL_SOCKET_IFNAME=ib0 -x NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_QPS_PER_CONNECTION=2 \
  -x NCCL_NET_GDR_LEVEL=PHB \
  -x DIST_INIT_ADDR=$(getent hosts $(head -n1 $PBS_NODEFILE) | awk '{print $1}') \
  bash -lc '
    export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
    outdir=${HOME}/scratch/nattanon/profiling/nsys_reports
    mkdir -p "$outdir"
    export PATH="/home/users/industry/ai-hpc/apacsc34/nsight-systems-2025.5.1/bin:$PATH"

    # === เลือกโหมด ===
    PROFILE_MODE="${PROFILE_MODE:-Paraver}"  # "Paraver"

    if [[ "$PROFILE_MODE" == "Paraver" ]]; then
      nsys profile \
        --gpu-metrics-devices=cuda-visible \
        -t cuda,nvtx \
        -o ${outdir}/DeepSeek_all_rank${OMPI_COMM_WORLD_RANK} \
        --capture-range=nvtx \
        --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
        --nvtx-capture=RANGE_NAME \
        ${HOME}/scratch/tanathep/py312/bin/python3 -m sglang.bench_offline_throughput \
          --model-path ${HOME}/scratch/model/DeepSeek-R1 \
          --dataset-path ${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
          --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
          --tp 8 --nnodes 2 --trust-remote-code \
          --dist-init-addr ${DIST_INIT_ADDR}:5000 \
          --node-rank ${OMPI_COMM_WORLD_NODE_RANK}

    fi

    # แปลงผลเป็น SQLite เพื่อสรุปอัตโนมัติได้ง่าย
    nsys export --sqlite true \
      -o ${outdir}/sglang_${PBS_JOBID}_rank${OMPI_COMM_WORLD_RANK}.sqlite \
      ${outdir}/sglang_${PBS_JOBID}_rank${OMPI_COMM_WORLD_RANK}.nsys-rep
  ' \
2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
