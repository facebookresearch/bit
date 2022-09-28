TASK_NAME=$1
GLUE_DIR=/data/glue_dataset/${TASK_NAME}
TEACHER_MODEL_DIR=/models/bert_base_uncased_pretrained_model/bert-base-uncased-${TASK_NAME}
STUDENT_MODEL_DIR=/models/bert_base_uncased_pretrained_model/bert-base-uncased-${TASK_NAME}
VOCAB_DIR=/models/bert_base_uncased_pretrained_model/bert-base-uncased-${TASK_NAME}
OUTPUT_DIR=./outputs

wbits=1
abits=1
JOB_ID=W${wbits}A${abits}
echo $TASK_NAME
echo $GLUE_DIR
echo $TEACHER_MODEL_DIR
echo $STUDENT_MODEL_DIR
echo $wbits
echo $abits
echo $JOB_ID

python quant_task_distill_glue.py \
    --data_dir ${GLUE_DIR} \
    --job_id ${JOB_ID} \
    --warmup_proportion 0.1 \
    --eval_step 100 \
    --ACT2FN relu \
    --output_dir ${OUTPUT_DIR}/${TASK_NAME}/${JOB_ID} \
    --distill_rep \
    --task_name $TASK_NAME \
    --teacher_model ${TEACHER_MODEL_DIR} \
    --student_model ${STUDENT_MODEL_DIR} \
    --vocab_dir ${VOCAB_DIR} \
    --weight_bits ${wbits} \
    --weight_quant_method bwn \
    --input_bits ${abits} \
    --input_quant_method elastic \
    --clip_lr 1e-4 \
    --weight_decay 0.01 \
    --learnable_scaling
