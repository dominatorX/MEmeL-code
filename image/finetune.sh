export CUDA_VISIBLE_DEVICES=0 # "0,1,2,3,4,5,6,7"
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_emel_finetune.yaml \
--pretrained ./checkpoints/swin_tiny_patch4_window7_224_22k.pth \
--data-path ../imagenet/imagenet/ --batch-size 64 --accumulation-steps 16 --use-checkpoint --emel
# if not open emel, delete --emel




















