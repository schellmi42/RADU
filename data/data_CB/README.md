# Cornell-Box Dataset

## Download

The CornellBox Dataset can be downloaded from this URL

>https://viscom.datasets.uni-ulm.de/radu/dataset.zip


## Dataset

The dataset contains correlation measurements, ToF depth images and ground truth depth images in `.hdr` format.

The script `simulate_noise_on_correlations.py` can be used to simulate shot noise on the correlation images using the default arguments.

## Citing this work

If you use this data in your work, please kindly cite the following paper:

```
@InProceedings{schelling2022radu,
author = {Schelling, Michael and Hermosilla, Pedro and Ropinski, Timo},
title = {{RADU} - Ray-Aligned Depth Update Convolutions for {ToF} Data Denoising},
booktitle = {Conference on Computer Vision and Patter Recognition (CVPR)},
year = {2022}
}
```

## References

The data was generated using the transient renderer of Jarabo et al. [1].

[1] Jarabo, A., Marco, J., Mu&#x00F1;oz, A., Buisan, R., Jarosz, W., Gutierrez, D.: "A framework for transient rendering" ACM Transactions on Graphics, SIGGRAPH ASIA, (2014). 