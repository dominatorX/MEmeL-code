We adopt the pytorch implementation of bert. Please clone the repo[https://github.com/dhlee347/pytorchic-bert.git] first. Then use the files given here to cover it, which contains modification for MEmeL.

In addition, we add configurations for the rest of datasets in Bert. 

Download pretrained model BERT-Base[https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip], Uncased and GLUE Benchmark Datasets[https://github.com/nyu-mll/GLUE-baselines] before fine-tuning.

The finetune and evaluation are similar to the repo[https://github.com/dhlee347/pytorchic-bert?tab=readme-ov-file#example-usage]. We provide a script "tune_eval.sh" to unify finetune and evaluation. One can edit the configs accordingly, the current file is an example for mrpc dataset.


