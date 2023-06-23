# Carla Offline RL Data Generation

This repository contains code for generating the data for training offline RL agents in the CARLA simulator.

For CARLA in docker:
```
docker run --privileged --gpus all --net=host -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.14 /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

For data generation:
```
cd D4RL
python scripts/generation/generate_carla.py 
```
