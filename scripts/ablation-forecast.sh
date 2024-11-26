CUDA_VISIBLE_DEVICES=1 wandb agent hzzheng/fatigue/31odno8l --count 25

python main.py forecast=ablation-Biking tag=forecast-Biking-manual forecast.model.d_layers=4 forecast.model.dropout=0.09995129150669854 forecast.train.bandwidth=2.545669950701619 forecast.train.mixup=0.6686797309310593 test=True
python main.py forecast=ablation-Stroop tag=forecast-Stroop-manual forecast.model.d_layers=3 forecast.model.dropout=0.05719786036755088 forecast.train.bandwidth=0.3781410652462365 forecast.train.mixup=0.8688551992145513 test=True
python main.py forecast=ablation-VR tag=forecast-VR-manual forecast.model.d_layers=4 forecast.model.dropout=0.023114709302635053 forecast.train.bandwidth=1.6866001164305633 forecast.train.mixup=0.6189883650285822
python main.py forecast=ablation-handgrip tag=forecast-handgrip-manual forecast.model.d_layers=3 forecast.model.dropout=0.06460644582375756 forecast.train.bandwidth=0.4830543128386242 forecast.train.mixup=0.6314615842515103