
# for task in powwer_nmse ideal_power_nmse; do
#     sbatch -J ${task} -o exp_${task}.log -p gpu-1080ti -c4 --gres gpu:1 submit.sh python train.py with nworker=4 loss_type=${task} opt_config.lr=0.2 report_interval=100 batch_size=16 ngpu=1 workdir=exp_${task}
# done

for task in dnnwpe_power_nmse; do
    sbatch -J ${task} -o exp_${task}.log -p gpu-1080ti -c4 --gres gpu:1 submit.sh ./train.py with nworker=4 workdir=exp_${task} loss_type=${task} batch_size=2 opt_config.lr=0.001
done
