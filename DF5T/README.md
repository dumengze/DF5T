# Diffusion-based foundation model for 5 task(DF5T)
 
## Online Demo



## User Interface for UniFMIR

1. Download the Finetuned Models

You can download the finetuned models and the examples of DF5T from [the release](https://drive.google.com/drive/folders/19gP4LV_GbyEcaHz3pufhUO5XVK9MFQgm?usp=drive_link). 


```
tasks/
    deno_em/
    sr2/
    deblur_em/
    inpaint_em/
    isotropic_em/
```

2. Install Packages

* Python 3.8
* Pytorch 2.0.1, CUDA 11.7 and CUDNN 
* Python Packages: 

You can install the required python packages by the following command:

```
conda env create -f environment.yaml
```

3. Run the Web Interface

You can run the interactive software with the following command:


```
python app.py
```

You can learn how to use the app through https://www.youtube.com/watch?v=2AW3lW8pVhw.

## Train


```
python /guided-diffusion-main/scripts/image_train.py
```


## Test

### 1. Prepare the datasets

1. All training and test data involved in the experiments are publicly available datasets. You can download our preprocessed data from [Science Data Bank, ScienceDB](https://www.scidb.cn/detail?dataSetId=8d30c6b23acd46d09e44114e8f739fe4) and unzip them into the corresponding folders. 


2. If your data is three-dimensional, preprocess it using the following code:


```
python pre/train.py
```

3. To preprocess 2D data, use the following code:


```
python pre/enhance.py
```
4. The dataset structure should be as follows:


```
exp
    datasets
            MitEM/MitEM
```
5. Use the following code to implement each task:


Denoise
```
python main.py --ni --config DF5T_512.yml --timesteps 50 --deg deno_em --sigma_0 0.2
```


Super resolution
```
python main.py --ni --config DF5T_512.yml --timesteps 50 --deg sr2 --sigma_0 0.2
```


Deblur
```
python main.py --ni --config DF5T_512.yml --timesteps 50 --deg deblur_em --sigma_0 0.2
```


2D-Inpaint
```
python main.py --ni --config DF5T_512.yml --timesteps 50 --deg inp_em --sigma_0 0.2
```


3D-Isotropic
```
python main.py --ni --config DF5T_512.yml --timesteps 50 --deg isotropic_em --sigma_0 0.2
```
