#!/bin/bash
# Define array of seed values
#SEEDS=(5 42 65 66 233)
SEEDS=(66)
loss_type_s=("ATL_MT")
for loss_type in "${loss_type_s[@]}"
do
  for SEED in "${SEEDS[@]}"
  do
    if [ "$loss_type" == "ATL_MT" ]; then
      learning_rate="4e-5"
      num_train_epochs="30"
      MT_lambda="3.5"
      loss_type_num="${MT_lambda}ATL_MT"
      num_class="100"
      rel2id_file_path="./meta/redocred_rel2id_4_Na.json"
      MT_Na="[(0, 11), (11, 32),(32, 79), (79, 100)]"
    fi

    python train.py \
        --data_dir "./dataset/redocred" \
        --transformer_type "bert" \
        --model_name_or_path "bert-base-cased" \
        --train_file "train_revised.json" \
        --dev_file "dev_revised.json" \
        --test_file "test_revised.json" \
        --train_batch_size 4 \
        --test_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --num_labels 4 \
        --learning_rate ${learning_rate} \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.06 \
        --num_train_epochs ${num_train_epochs} \
        --seed ${SEED} \
        --loss_type ${loss_type} \
        --save_name "redocred_bert_${loss_type_num}_${num_train_epochs}epoch_${learning_rate}_seed${SEED}" \
        --load_path "./checkpoint/redocred_bert_${loss_type_num}_${num_train_epochs}epoch_${learning_rate}_seed${SEED}.pt" \
        --MT \
        --num_class ${num_class} \
        --rel2id_file_path ${rel2id_file_path} \
        --MT_Na "${MT_Na}" \
        --MT_lambda ${MT_lambda} \
        --cuda_device 0
  done
done
