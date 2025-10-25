#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=2:mem=1880gb
#PBS -j oe
#PBS -m abe

module load cuda

export DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE)

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile ${PBS_NODEFILE} \
  -map-by ppr:1:node:PE=112 -oversubscribe -use-hwthread-cpus \
  -bind-to none --report-bindings -display-map \
  -tag-output -output-filename ${HOME}/scratch/tanathep/run/sglang.${PBS_JOBID} \
  -x PATH -x NCCL_DEBUG=INFO -x DIST_INIT_ADDR \
  bash -lc '
    RANK=${OMPI_COMM_WORLD_RANK}
    LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
    NODE_RANK=$(( RANK / ${OMPI_COMM_WORLD_LOCAL_SIZE:-2} ))

    OUTDIR=${HOME}/scratch/nattanon/profiling/nsys_reports
    mkdir -p $OUTDIR

    if [ "$LOCAL_RANK" -eq 0 ]; then
      echo "[INFO] Rank $RANK (Node $NODE_RANK, LocalRank $LOCAL_RANK) will be profiled."
      /home/users/industry/ai-hpc/apacsc34/nsight-systems-2025.5.1/bin/nsys profile \
        --output=$OUTDIR/sglang_profile.${PBS_JOBID}.node${NODE_RANK}.rank${RANK} \
        --trace=cuda,nvtx,mpi,osrt \
        --capture-range=nvtx --capture-range-end=stop --sample=none --force-overwrite=true \
        ${HOME}/scratch/nattanon/py312/bin/python3 -m sglang.bench_offline_throughput \
          --model-path ${HOME}/scratch/model/DeepSeek-R1 \
          --dataset-path ${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
          --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
          --tp 16 --nnodes 2 --trust-remote-code \
          --dist-init-addr ${DIST_INIT_ADDR}:5000 \
          --node-rank ${NODE_RANK}
    else
      echo "[INFO] Rank $RANK (Node $NODE_RANK, LocalRank $LOCAL_RANK) runs without profiling."
      ${HOME}/scratch/nattanon/py312/bin/python3 -m sglang.bench_offline_throughput \
        --model-path ${HOME}/scratch/model/DeepSeek-R1 \
        --dataset-path ${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
        --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
        --tp 16 --nnodes 2 --trust-remote-code \
        --dist-init-addr ${DIST_INIT_ADDR}:5000 \
        --node-rank ${NODE_RANK}
    fi
  ' \
  2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
