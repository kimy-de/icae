[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6817442.svg)](https://doi.org/10.5281/zenodo.6817442)
# Convolutional Autoencoders and Clustering for Low-dimensional Parametrization of Incompressible Flows

## 1. Publication

[Jan Heiland and Yongho Kim (2022) Convolutional Autoencoders and Clustering for Low-dimensional Parametrization of Incompressible Flows, 25th International Symposium on Mathematical Theory of Networks and Systems (MTNS), IFAC-PapersOnLine](https://www.sciencedirect.com/science/article/pii/S2405896322027240) 

The design of controllers for general nonlinear PDE models is a diffcult task because of the high dimensionality of the partially discretized equations. It has been observed that the embedding of nonlinear systems into the class of linear parameter varying systems (LPV) gives way to apply linear theory and methods from numerical linear algebra for controller design. The feasibility of the LPV approach hinges on the dimension of the inherent parametrization. 
In this work we propose and evaluate combinations of convolutional neural networks and clustering algorithms for very low-dimensional parametrizations of incompressible Navier-Stokes equations.

## 2. Overview

* Traning session:  
    * `train.py` -> `centroids.py` -> `icae_train.py`
    * CAEs: 5 cases (reduced dimension [2,3,5,8,12]) 
    * k-means clustering: 15 cases (reduced dimension [2,3,5,8,12] x number of clusters [5,30,100])
    * iCAEs: 10 cases (reduced dimension [2,3,5,8,12] x number of clusters [5,30])
* Evaluation session:
    * Reconstruction errors: `evaluation.py` (POD, CAE, CAE100, iCAE5, iCAE30)
    * Approximation graphs: `trajectory.py` (POD, CAE, iCAE30)
    * 2-,3-dimensional $\rho$ distributions: `dist2d3d.py` (CAE)
* Directory information:
    * data: train data, test data, and POD-mode data
    * models: pretrained models and centroid data
    * results: result images
* Used libraries:
    * os, argparse, tqdm, time, matplotlib, numpy, sklearn, torch

* We provide everything including our pretrained models (pretrain.zip) and data. You can check the results without retraining after decompressing pretrain.zip in the "models" folder.

* If you want to compute everything from scratch (note that this may take several hours), use
```sh
source runitall.sh
```



## 3. Datasets and pretrained models

* Reynolds number: 40
* Train data: 400 snapshots in [0,10]
* Evaluation data: 800 snapshots in [0,10]
* Snapshot size $n_v$: 5922 (2x47x63)

The datasets and pretrained models are available via [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6817442.svg)](https://doi.org/10.5281/zenodo.6817442).

## 4. Training session

#### 4.1. CAEs

* batch_size: 64
* numer of epoch: 4000 
* latent variable dimension $n_\rho$: 2, 3, 5, 8, 12
* learning rate: 1e-3 (1e-4 if $n_\rho=2$)

```python
python train.py --latent_size 12 --num_epochs 3000 # --latent_size 3, 5, 8
```
```python
python train.py --latent_size 2 --num_epochs 4000  --lr 1e-4
```

#### 4.2. k-means clustering

```python
python centroids.py
```

#### 4.3. iCAEs

* number of epoch: 15000 
* latent variable dimension $n_\rho$: 2, 3, 5, 8, 12
* number of clusters $k$: 5, 30
* learning rate: 1e-3 

```python
python icae_train.py --latent_size 12 --num_clusters 5 # --latent_size 2, 3, 5, 8
```


## 5. Reproducibility of experimental results

* All the pretrained models and centroid data must be prepared.

#### 5.1. Reconstruction errors

```python
python evaluation.py
```

#### 5.2. Sample trajectories

```python
python trajectory.py --latent_size 2 # 3, 5, 8, 12
```

#### 5.3. 2-,3-dimensional $\rho$ distributions
```python
python dist2d3d.py
```
