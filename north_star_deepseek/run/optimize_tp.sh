#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=2:mem=1880gb
#PBS -j oe
#PBS -m abe

module load cuda

time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
 -hostfile ${PBS_NODEFILE} \
 -map-by ppr:1:node:PE=112 -oversubscribe -use-hwthread-cpus \
 -bind-to none --report-bindings -display-map \
 -tag-output -output-filename ${HOME}/scratch/phattadon/optimize/optimize_dp2.${PBS_JOBID} \
 -x PATH \
 -x NCCL_DEBUG=INFO \
 -x DIST_INIT_ADDR=$(head -n 1 $PBS_NODEFILE) \
 bash -c 'time nsys profile \
 --output=${HOME}/scratch/phattadon/profiling/nsys_reports/sglang_profile.${PBS_JOBID}.${OMPI_COMM_WORLD_RANK} \
 --trace=cuda,nvtx,mpi,osrt \
 ${HOME}/scratch/phattadon/py312/bin/python3 \
 -m sglang.bench_offline_throughput \
 --model-path ${HOME}/scratch/model/DeepSeek-R1 \
 --dataset-path ${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json \
 --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
 --tp 16 --dp 2 --nnodes 2 --trust-remote-code \
 --dist-init-addr ${DIST_INIT_ADDR}:5000 --node-rank ${OMPI_COMM_WORLD_RANK}' \
 2>&1 | tee ${HOME}/run/stdout.sglang.${PBS_JOBID}
apacsc34@a2ap-login02:~/scratch/phattadon/profiling$

