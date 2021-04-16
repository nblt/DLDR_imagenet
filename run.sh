# CUDA_VISIBLE_DEVICES=0,1 python pca.py --n_components 60
for model in resnet18
do
CUDA_VISIBLE_DEVICES=0,1,2  python3 main_lt.py -a $model --epochs 90  --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/pami/Datasets/ILSVRC2012
#CUDA_VISIBLE_DEVICES=0,1,2 python pca.py --n_components 50 --epochs 50 --params_start 1 --params_end 301
#CUDA_VISIBLE_DEVICES=1,2,0 python -u train_cuda.py --epochs 50 --accumulate_grad 1 --alpha 0.00 --params_start 1 --params_end 301  --batch-size 256  --n_components 50 --arch=$model  --save-dir=save_$model |& tee -a log_$model 

done



