# Human performance forecasting with transformers


## Set up environment
For convenience, [Dockerfile](Docker/Dockerfile) is provided under `Docker`. 
You can use as follows:
```bash
# Build docker image
docker build -t [image tag] --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .

# Run docker container
docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v [path to the top of this git repo]:/workdir -v [path to data]:/data [image tag]
```
Breakdown of the `docker run` command:
- `--gpus all -it --rm`: With all GPUs enabled, run an interactive session, and automatically remove the container when it exits.
- `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`: Flags recommended by NVIDIA. Unlock the resource constraint.
- `-v [path to the top of this repo]:/workdir -v [path to data]:/data`: Mount the current dir to `/workdir`. Mount the data directory to `/data`.


## Classification
```bash
python main.py tag=train-best task=classification
```
Evaluate classifier, 
```bash
python main.py tag=train-best task=classification test=True
```

## Fatigue prediction
```bash
python main.py forecast=vae250-Biking tag=forecast-Biking-best forecast.model.d_layers=4 forecast.model.dropout=0.08101762186968295 forecast.train.bandwidth=1.1351408305181596 forecast.train.mixup=0.8996279310287609
```


## Pretrain VAE on EMG & Pulse
```bash
bash scripts/encode.sh
```


## Hyperparameter sweep

