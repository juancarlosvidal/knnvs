#!/bin/bash
#SBATCH -c 16
#SBATCH --mem=128gb
#SBATCH --array=1-22%8

# task_id -> "script  output_csv  output_dir"
declare -A JOB
JOB[1]="code/low_1.py    sim1.csv      sim/low/data"
JOB[2]="code/low_1_no.py sim1_no.csv   sim/low/data"
JOB[3]="code/low_2.py    sim2.csv      sim/low/data"
JOB[4]="code/low_2_no.py sim2_no.csv   sim/low/data"
JOB[5]="code/low_3.py    sim3.csv      sim/low/data"
JOB[6]="code/low_3_no.py sim3_no.csv   sim/low/data"
JOB[7]="code/med_1.py    med1.csv      sim/moderate/data"
JOB[8]="code/med_1_no.py med1_no.csv   sim/moderate/data"
JOB[9]="code/med_2.py    med2.csv      sim/moderate/data"
JOB[10]="code/med_2_no.py med2_no.csv  sim/moderate/data"
JOB[11]="code/med_3.py    med3.csv     sim/moderate/data"
JOB[12]="code/med_3_no.py med3_no.csv  sim/moderate/data"
JOB[13]="code/med_4.py    med4.csv     sim/moderate/data"
JOB[14]="code/med_4_no.py med4_no.csv  sim/moderate/data"
JOB[15]="code/lar_1.py    lar1.csv     sim/large/data"
JOB[16]="code/lar_1_no.py lar1_no.csv  sim/large/data"
JOB[17]="code/lar_2.py    lar2.csv     sim/large/data"
JOB[18]="code/lar_2_no.py lar2_no.csv  sim/large/data"
JOB[19]="code/lar_3.py    lar3.csv     sim/large/data"
JOB[20]="code/lar_3_no.py lar3_no.csv  sim/large/data"
JOB[21]="code/lar_4.py    lar4.csv     sim/large/data"
JOB[22]="code/lar_4_no.py lar4_no.csv  sim/large/data"

read -r SCRIPT OUTFILE OUTDIR <<< "${JOB[$SLURM_ARRAY_TASK_ID]}"
mkdir -p "$OUTDIR"

if [ -f "$OUTDIR/$OUTFILE" ]; then
    echo "task=$SLURM_ARRAY_TASK_ID script=$SCRIPT -> $OUTDIR/$OUTFILE already exists, skipping"
    exit 0
fi

PYTHON=/mnt/beegfs/home/juan.vidal/miniforge3/envs/knnvs/bin/python3.8
echo "task=$SLURM_ARRAY_TASK_ID script=$SCRIPT -> $OUTDIR/$OUTFILE"
$PYTHON $SCRIPT && mv "$OUTFILE" "$OUTDIR/"
