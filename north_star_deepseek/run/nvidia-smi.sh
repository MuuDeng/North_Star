#!/bin/bash
#PBS -P 50000128
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=112:ngpus=8:mpiprocs=8:mem=1880gb
#PBS -j oe
#PBS -m abe

nvidia-smi
