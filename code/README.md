![ov](https://github.com/zyainfal/Joint-Holistic-and-Masked-Face-Recognition/blob/main/fig/overview.PNG)

Official implements of paper ``[Joint Holistic and Masked Face Recognition](https://ieeexplore.ieee.org/abstract/document/10138097)''

# Enviroment
Please check `env.yml` to build the enviroment by anaconda3.

# How to use
## Dataset
Training data (MS1MV2, with masked faces) and evaluation data (LFW,  AGE-DB, CFP-FP, SMFRD , and RMFVD) are zipped [here](https://drive.google.com/file/d/1nR1gd9u4LxntACMe50RfgFfPjRmrtgYT/view?usp=sharing).

## Training/Evaluation
We provide training code for training both FaceT-B and IR100 on MS1MV2, and evaluation code during the training for LFW, AGE-DB, CFP-FP, SMFRD (masked LFW, AGE-DB, and CFP-FP), and RMFVD.

All masked faces are aligned by RetinaFace, while some evaluation faces are poorly aligned due to heavily occlusion. 

Here are some examples for training the model:


```
# Pre-training FaceT-B by MAE:
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 pretrainer.py --model facet_base --ntype deepnorm --drop_path 0.0 --data_path /data1/yuhao.zhu/data/faces_emore/ --train_type mae --epochs 10 --warmup_epochs 2 --output_dir ./workspace/ckpts/pretrain/mae_facet_base_ms1mv2_deepnorm --log_dir ./workspace/logs/pretrain/mae_facet_base_ms1mv2_deepnorm --batch_size 512 --opt adamw --opt_betas 0.9 0.95 --clip_grad 1.0 --fp16
```


```
# Fine-tuning FaceT-B by ArcFace/CurricularFace/MagFace (learning rate for training on WebFace42M is 0.00075):
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 trainer.py --model facet_base --ntype deepnorm --drop_path 0.1 --head magface --transform_layer 12 --data_path /data1/yuhao.zhu/data/faces_emore/  --output_dir ./workspace/ckpts/train/mae_facet_base_deepnorm_magface_deepnorm_ms1mv2_m10_p12_n32_pp_e10 --log_dir ./workspace/logs/train/mae_facet_base_deepnorm_magface_deepnorm_ms1mv2_m10_p12_n32_pp_e10 --opt adamw --opt_betas 0.9 0.999 --lr 0.001 --weight_decay 0.05 --warmup_epochs 1 --epochs 10 --batch_size 256 --fp16 --resume /data1/yuhao.zhu/projects/facet/workspace/ckpts/pretrain/mae_facet_base_ms1mv2_deepnorm/checkpoint-last.pth --clip_grad 1.0 --eval_freq 3000 --layer_decay 0.75 --masked_faces 0.1
```


```
# Training IR100 from scartch:
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 trainer.py --model ir100 --head magface --data_path /data1/yuhao.zhu/data/faces_emore/  --output_dir ./workspace/ckpts/train/ir100_deepnorm_magface_deepnorm_ms1mv2_m10_e20 --log_dir ./workspace/logs/train/ir100_deepnorm_magface_deepnorm_ms1mv2_m10_e20 --opt sgd --momentum 0.9 --lr 0.025 --weight_decay 5e-4 --warmup_epochs 1 --epochs 20 --batch_size 768 --layer_decay 1 --fp16 --masked_faces 0.1 --eval_freq 3000
```

## Trained Model
Both [FaceT-B](https://drive.google.com/file/d/1d0q1NbDUISDjbE4Gsyl6_B9Tj5wBbaGT/view?usp=sharing) and [IR100](https://drive.google.com/file/d/1cy71hnq8N5WZ3B0o8U69O80qkjZi1alL/view?usp=sharing) trained on WebFace42M (with 10% masked faces) are provided in Google Drive.

MAE pre-trained FaceT-B is [here](https://drive.google.com/file/d/1H_iY_uEeQ_MkpYI4nckjhc5NNJxeG8SV/view?usp=sharing).
