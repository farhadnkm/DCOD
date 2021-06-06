# Deep Compressive Object Decoder (DCOD)
Reconstruct in-line holograms merely with a single image. In the proposed model, a regularized untrained deep neural network inversely generates and optimizes the object field based on the acquired hologram.

Check out our paper and results:

https://www.nature.com/articles/s41598-021-90312-5

All experiments can be found in the **DCOD_Implementation.ipynb**

## Requirements

To make the workflow expandable and easy to implement, holographic reconstruction algorithms and data processing tools are encapsulated in a python package named ***Fringe*** (https://github.com/farhadnkm/Fringe.Py). 

To install this package, run:

```
pip install fringe
```

This project also requires ***tensorflow-addons***.

## Notice:

If you are getting error on hologram import due to LWZ compression, do the following:

```
pip uninstall tifffile
pip install imagecodecs imagecodecs-lite
pip install tifffile
```
