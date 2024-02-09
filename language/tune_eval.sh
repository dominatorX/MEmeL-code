export task=mrpc  # "mnli" "CoLA" "mrpc" "QNLI" "RTE" "SST2" "STSB" "QQP"
export emel=emel  # emel or empty ""
export SAVE_DIR=/save/dir
export BERT_PRETRAIN=/bert/pretrain/file
export GLUE_DIR=/glue/data/dir
export CUDA_VISIBLE_DEVICES=0

for lr in "2e-5" "3e-5" "4e-5" "5e-5"
    do
      mkdir $SAVE_DIR"$lr"
      python classify.py \
          --task $task \
          --mode "$emel"_train \
          --train_cfg config/train_"$task".json \
          --lr $lr \
          --model_cfg config/bert_base.json \
          --data_file $GLUE_DIR/$task/train.tsv \
          --pretrain_file $BERT_PRETRAIN/bert_model.ckpt \
          --vocab $BERT_PRETRAIN/vocab.txt \
          --save_dir $SAVE_DIR"$lr" \
          --max_len 128

  for test_file in "dev.tsv"
  do
      python classify.py \
                  --task $task \
                  --mode eval \
                  --train_cfg config/train_"$task".json \
                  --model_cfg config/bert_base.json \
                  --data_file $GLUE_DIR/$task/$test_file \
                  --model_file $SAVE_DIR"$lr"/model_steps_345.pt \
                  --vocab $BERT_PRETRAIN/vocab.txt \
                  --max_len 128
  done
done

