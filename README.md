# TI Estimation

Official source code for paper 《TFG-Net: Tropical Cyclone Intensity Estimation from a Fine-grained Perspective with the Graph Convolution Neural Network》

The implementation of the FTFE module is referred to this [repository](https://github.com/jeong-tae/RACNN-pytorch). Many thanks to the contributor [@jeong-tae](https://github.com/jeong-tae)

### Overall Architecture of TFG-Net
![image](https://github.com/xuguangning1218/TI_Estimation/blob/master/figure/model.png)

### Auto Detect Fine-grained Tropical Cyclone Feature during Training Stage
![image](https://github.com/xuguangning1218/TI_Estimation/blob/master/figure/TFG-Net_training%402x.gif)

### Environment Installation
```
conda env create -f TI_Estimation.yaml
```  

### Data Preparation 
* Download the required GridSat dataset from NOAA official site through [here](<https://www.ncei.noaa.gov/products/gridded-geostationary-brightness-temperature> "here") and the required tropical cyclone best track dataset from NOAA official site through [here](<https://www.ncdc.noaa.gov/ibtracs/>  "here"). 
* Or you can download the preprocessing GridSat data from my google drive through [here](<https://drive.google.com/drive/folders/1-4xPJxZEaofC1vJfKwK10Iwi9ocueFLZ?usp=sharing> "here"). Note that the ibtracs tropical cyclone best track dataset is provided in folder ***data***.

###  Reproducibility
We provide one of the five runs best-validated models in [here](<https://drive.google.com/drive/folders/1-FGSiIMvJm0XlLq0rOf_-_3-rl-FouBz?usp=sharing>  "here").  You can reproduce the result reported in the paper using this best-validated model.


###  Source Files Description

```
-- data # dataset folder
  -- GridSat_B1_new_npy # the GridSat data folder. You need to download it from google driver 
  -- gridsat.img.min.max.npy # the min and max value of the training GridSat dataset 
  -- gridsat.path.ibtr.windspeed.csv # the label GridSat file with satellite images save path 
  -- GridSat_B1_processor.ipynb # the orignal nc GridSat file processor 
  -- GridSAT_invalid_img.ipynb # the invalid GridSat preprocessor 
  -- ibtracs.ALL.list.v04r00.rar # the rar compression file of the IBTracs tropical cyclone best track dataset 
-- figure # figure provider
  -- network.png # architecture of TFG-Net model 
  -- TFG-Net_training@2x.gif # the training stage fine-grained tropical feature captures with 20 epochs interval
-- layers # necessary layer
  -- AttentionCrop.py # the Attention Cropper
  -- GraphConvolution.py # the Graph Convolution
  -- MultiHeadGAT.py # Multi Head GAT
-- model_saver # model save path
  -- best_validate_model.pth # best model (one of five runs). You need to download it from google driver
  -- TFG-Net.log # the training loss of the TFG-Net
TI_Estimation.yaml # conda environment for the project
TFG-Net.ipynb # jupyter visualized code for the TFG-Net
```

### Run

When the conda environment and datasets are ready, you can train or reproduce our result by running the file `TFG-Net.ipynb`.

### Citation
If you find this code or idea useful, please cite our work:
```bib
@inproceedings{xu2022TFG,
  title={TFG-Net: Tropical Cyclone Intensity Estimation from a Fine-grained Perspective with the Graph Convolution Neural Network},
  author={Xu, Guangning and Yan Li and Chi Ma and Xutao Li and Yunming Ye and Qingquan Lin and Zhichao Huang and Shidong Chen},
  JO={Engineering Applications of Artificial Intelligence},
  volume={In Press},
  year={2022}
}
```
