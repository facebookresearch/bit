TASK_NAME=squad
DATA_DIR=/data/squad_dataset
TEACHER_MODEL_DIR=/models/bert-base-uncased/squad-v1
STUDENT_MODEL_DIR=/models/bert-base-uncased/squad-v1
OUTPUT_DIR=./outputs

wbits=1
abits=1
JOB_ID=W${wbits}A${abits}
echo $TASK_NAME
echo $DATA_DIR
echo $TEACHER_MODEL_DIR
echo $STUDENT_MODEL_DIR
echo $wbits
echo $abits
echo $JOB_ID



python quant_task_distill_squad.py \
    --data_dir ${DATA_DIR} \
    --job_id ${JOB_ID} \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --warmup_proportion 0.2 \
    --max_seq_len 384 \
    --eval_step 500 \
    --num_train_epochs 3 \
    --ACT2FN relu \
    --output_dir ${OUTPUT_DIR}/${TASK_NAME}/${JOB_ID} \
    --task_name $TASK_NAME \
    --teacher_model ${TEACHER_MODEL_DIR} \
    --student_model ${STUDENT_MODEL_DIR} \
    --weight_bits ${wbits} \
    --weight_quant_method bwn \
    --input_bits ${abits} \
    --input_quant_method elastic\
    --clip_lr 1e-4 \
    --weight_decay 0.01 \
    --learnable_scaling \
