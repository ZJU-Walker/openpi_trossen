# Cluster scripts

Slurm batch scripts for training on the Iris cluster.

## Submit a job

```bash
cd /iris/u/kewalk/openpi_trossen
mkdir -p logs/human_robot_pack_full_0514   # one-time, matches sbatch --output path
sbatch cluster_scripts/train_full_slurm.sbatch
```

Returns `Submitted batch job <JOBID>`.

## Check status

```bash
squeue --me                                  # all my jobs (PD=pending, R=running)
squeue -j <JOBID>                            # one specific job
scontrol show job <JOBID>                    # full details (node, resources, time)
sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode    # post-run summary
```

## Watch logs

```bash
tail -f logs/human_robot_pack_full_0514/pi05_human_robot_pack_full_0514-<JOBID>.out
tail -f logs/human_robot_pack_full_0514/pi05_human_robot_pack_full_0514-<JOBID>.err
```

## Cancel

```bash
scancel <JOBID>
```

## Current scripts

- `train_full_slurm.sbatch` — full-parameter finetune of pi0.5 on `pack_with_human`.
  Runs `compute_norm_stats.py` then `train.py` on a single H100 (`iris-hgx-1` or `iris-hgx-2`). Config: `pi05_trossen_pack_with_human_full`, exp-name: `pi05_pack_with_human_full_0514`. Wall time: 1 day.

## Check node availability

```bash
squeue -w iris-hgx-1 \
  -o "%.10i %.12u %.18P %.30j %.2t %.10M %.6D %.20R %.20b"
```

## Attach an interactive terminal inside that job with:

```bash
srun --jobid=15448830 --overlap --pty bash
```