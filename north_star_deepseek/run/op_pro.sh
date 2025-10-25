#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=06:00:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=2:mem=1880gb
#PBS -j oe
#PBS -m abe

# ========= 0) โมดูลพื้นฐาน =========
module load cuda
module load ninja/1.12.1

# ========= 1) ค่าพื้นฐาน/ลด overhead =========
export OMP_NUM_THREADS=8                       # พอเหมาะกับ 1 rank/GPU
export TOKENIZERS_PARALLELISM=false            # กันสลับเธรดเกินจำเป็น
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY                # ลดเวลาโหลด PTX/CUBIN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHOW_CPP_STACKTRACES=0

# ระบุสถาปัตย์ลดเวลาคอมไพล์ extension (ปรับตามเครื่อง: A100=8.0, H100=9.0)
export TORCH_CUDA_ARCH_LIST="9.0"

# ========= 2) NCCL/UCX tuning เพื่อลดคอมมู =========
# หมายเหตุ: ค่า iface/HCA/GID ด้านล่างต้องให้ตรงกับคลัสเตอร์จริง
export NCCL_DEBUG=WARN                         # เปิดตอนดีบัก: INFO, ปิดจริง: WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="^lo,docker0"
export NCCL_IB_HCA="mlx5_0,mlx5_1"
# ถ้า RoCEv2: GID index มัก 2 หรือ 3; ถ้า IB ล้วน อาจเป็น 0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=PHB                  # หรือ PIX/NVL ตาม topology
export NCCL_MIN_NCHANNELS=8
export NCCL_MAX_NCHANNELS=32
# โปรโตคอล/อัลกอริทึม: ลองสลับ Ring/Tree และ Simple/LL/LL128 ตามขนาดข้อความจริง
export NCCL_ALGO=Ring
export NCCL_PROTO=LL128
# เปิด CollNet (ช่วยกรณีหลายโหนดบาง topology)
export NCCL_COLLNET_ENABLE=1

# ใช้ UCX path ของ OpenMPI ให้คุยได้ดีขึ้นบน IB/RoCE
export OMPI_MCA_pml=ucx
export OMPI_MCA_osc=ucx
export UCX_TLS=rc,tcp,gdr,sm,self
export UCX_NET_DEVICES=all

# ========= 3) จุด rendezvous สำหรับ torch.distributed =========
export MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
export MASTER_PORT=$((20000 + RANDOM % 10000))

# ========= 4) พาธไฟล์และพารามิเตอร์เบนช์ =========
PYTHON="${HOME}/scratch/tanathep/py312/bin/python3"
MODEL="${HOME}/scratch/model/DeepSeek-R1"
DATA="${HOME}/scratch/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

# ========= 5) รัน 16 ranks = 8/โหนด (TP=8 ภายในโหนด) =========
time /usr/mpi/gcc/openmpi-4.1.7a1/bin/mpirun \
  -hostfile "${PBS_NODEFILE}" \
  -map-by ppr:1:node:PE=112 -bind-to core \
  --report-bindings -display-map -tag-output \
  -x PATH \
  -x MASTER_ADDR -x MASTER_PORT \
  -x OMP_NUM_THREADS -x TOKENIZERS_PARALLELISM \
  -x CUDA_DEVICE_ORDER -x CUDA_MODULE_LOADING -x PYTORCH_CUDA_ALLOC_CONF \
  -x TORCH_CUDA_ARCH_LIST \
  -x NCCL_DEBUG -x NCCL_ASYNC_ERROR_HANDLING -x NCCL_SOCKET_IFNAME -x NCCL_IB_HCA -x LD_LIBRARY_PATH -x CUDA_HOME \
  -x NCCL_IB_GID_INDEX -x NCCL_NET_GDR_LEVEL -x NCCL_MIN_NCHANNELS -x NCCL_MAX_NCHANNELS \
  -x NCCL_ALGO -x NCCL_PROTO -x NCCL_COLLNET_ENABLE \
  -x OMPI_MCA_pml -x OMPI_MCA_osc -x UCX_TLS -x UCX_NET_DEVICES \
  bash -lc "
    set -euo pipefail
    RANK=\${OMPI_COMM_WORLD_RANK}
    LOCAL_RANK=\${OMPI_COMM_WORLD_LOCAL_RANK}
    LOCAL_SIZE=\${OMPI_COMM_WORLD_LOCAL_SIZE:-8}
    NODE_RANK=\$(( RANK / LOCAL_SIZE ))

    # map 1 rank → 1 GPU
    export CUDA_VISIBLE_DEVICES=\${LOCAL_RANK}

    echo \"[DBG] host=\$(hostname) rank=\${RANK} local=\${LOCAL_RANK} node=\${NODE_RANK} world=\${OMPI_COMM_WORLD_SIZE}\" 1>&2

    ${PYTHON} -m sglang.bench_offline_throughput \
      --model-path ${MODEL} \
      --dataset-path ${DATA} \
      --num-prompts 2000 --load-format dummy --seed 2025 --dtype bfloat16 \
      --tp 8 --pp 2 --nnodes 2 --trust-remote-code \
      --dist-init-addr \${MASTER_ADDR}:\${MASTER_PORT} \
      --node-rank \${NODE_RANK} \
  " 2>&1 | tee "${HOME}/run/stdout.sglang.${PBS_JOBID}"
