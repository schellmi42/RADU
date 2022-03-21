# RADU - Ray-Aligned Depth Update Convolutions for ToF Data Denoising

### [Arxiv](https://arxiv.org/abs/2111.15513)
> RADU - Ray-Aligned Depth Update Convolutions for ToF Data Denoising <br />
> [Michael Schelling](https://www.uni-ulm.de/?id=michael-schelling), [Pedro Hermosilla](https://www.uni-ulm.de/?id=pedro-hermosilla-casajus), [Timo Ropinski](https://www.uni-ulm.de/in/mi/institut/mitarbeiter/timo-ropinski/) <br />
> Conference on Computer Vision and Patter Recognition (CVPR) - 2022



This repository contains the TensorFlow 2 code the for the RADU network.

The code was tested using TensorFlow 2.3.0 and Python 3.6.9 on Ubuntu 18.

## Dockerfile

To setup the environment it is advised to use the following dockerfile
```
FROM tensorflow/tensorflow:2.3.0-gpu
	
RUN apt-get update
RUN apt-get -y install ffmpeg libsm6 libxext6 git

RUN git clone https://github.com/schellmi42/tensorflow_graphics_point_clouds /pclib
RUN git clone https://github.com/schellmi42/graphics /tfg
RUN git clone https://github.com/schellmi42/RADU /RADU
RUN pip install -r /RADU/requirements.txt
RUN export PYTHONPATH="$/pclib:/tfg/graphics:$PYTHONPATH";
RUN python -c 'import imageio; imageio.plugins.freeimage.download()'

WORKDIR /RADU
```

Installation of [NVIDIA-Docker-Support](https://github.com/NVIDIA/nvidia-docker) is necessary.

To create the docker image run the following (sudo) in the location you pasted the `Dockerfile`
```
nvidia-docker build -t radu .
```
Start the docker container using the  `nvidia-container-toolkit` and `--gpus all` flags.


## Dataset

### Cornell-Box Dataset

The Cornell-Box Dataset can be downloaded from this URL

>https://viscom.datasets.uni-ulm.de/radu/dataset.zip

More information about the dataset is available in [data/data_CB](data/data_CB).

### External Datasets

The datasets from Agresti et al [1] are available at this URL:

>https://lttm.dei.unipd.it/paper_data/MPI_DA_CNN/

The FLAT dataset [2] is available at this GIT repository

>https://github.com/NVlabs/FLAT

#### Loading of the Datasets

Following the data structure provided in [data/](data/), when placing the datasets to prevent errors during loading.

To load the datasets in a docker container it is advised to mount the data folders into the container at `/RADU/data/data_*/` using the `docker --volume` flag.

The paths to the datasets may also be specified indiviually in the `DATA_PATH` variable inside the respective `data_loader.py` files.

## Pretrained model weights

Pretrained model weights of the RADU Network on the Datasets S1&S2, the FLAT dataset and the CB-Dataset are available at this URL:

> https://viscom.datasets.uni-ulm.de/radu/trained_weights.zip

To evaluate the network using the pretrained weights use the following commands:

On the real datasets S4 and S5 [1] 
```
python code_dl/eval_RADU_NN.py --d data_agresti/S4 --l trained_weights/Agresti/U-DA/ --skip_3D --feature_type mf_agresti
python code_dl/eval_RADU_NN.py --d data_agresti/S5 --l trained_weights/Agresti/U-DA/ --skip_3D --feature_type mf_agresti
```

On the Cornell-Box dataset

```
python code_dl/eval_RADU_NN.py --d data_CB --l trained_weights/CBDataset --skip_3D --feature_type mf_agresti
```

On the FLAT dataset [2]

```
python code_dl/eval_RADU_NN.py --d data_FLAT --l trained_weights/FLAT --skip_3D --feature_type mf_agresti
```
## Citing this work

If you use this code in your work, please kindly cite the following paper:

```
@InProceedings{schelling2022radu,
author = {Schelling, Michael and Hermosilla, Pedro and Ropinski, Timo},
title = {{RADU} - Ray-Aligned Depth Update Convolutions for {ToF} Data Denoising},
booktitle = {Conference on Computer Vision and Patter Recognition (CVPR)},
year = {2022}
}
```

## References

[1]  G. Agresti, H. Schaefer, P. Sartor, P. Zanuttigh: "Unsupervised Domain Adaptation for ToF Data Denoising with Adversarial Learning", CVPR, (2019). 

[2] Q. Guo, I. Frosio, O. Gallo, T. Zickler J. Kautz: "Tackling 3D ToF Artifacts Through Learning and the FLAT Dataset", ECCV, (2018).
