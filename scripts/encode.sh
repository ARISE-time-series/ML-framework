# pretrain EMG encoder
python3 train_vae.py --config configs/pretrain/EMG-vae.yaml

# pretrain Pulse encoder
python3 train_vae.py --config configs/pretrain/Pulse-vae.yaml

# encode EMG
python3 encode_data.py --config configs/pretrain/EMG.yaml \
--ckpt exps/encoder/EMG-encoder-192modes/checkpoints/ckpt_100000.pt --outdir ../data/fatigue

# encode Pulse data
python3 encode_data.py --config configs/pretrain/Pulse.yaml \
--ckpt exps/encoder/Pulse-encoder-32modes/checkpoints/ckpt_100000.pt --outdir ../data/fatigue