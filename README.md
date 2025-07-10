# PhD_Code_Zhengyang_LAN
This repository contains the code of Zhengyang LAN's work during his PhD.

Based on the PhD research, it is divided into three parts:
- Deep learning-based 3DGK diagnostic tool (Paper: [Towards a diagnostic tool for neurological gait disorders in childhood combining 3D gait kinematics and deep learning](https://www.sciencedirect.com/science/article/pii/S0010482524001793))
- XAI + Diagnostic Tools Abstract:[Analysis Of Deep Learning Classification Methods Of
3D Kinematic Data For Neurological Gait Disorders](https://brgm.hal.science/LAB-STICC_T2I3/hal-04576157v1)
- XAI evaluation (quantitative analysis) (Paper come soon)

## Install & Environment

This implementation is Python 3.9.11 on a PC running the Ubuntu 24.04.2 LTS operating system. In theory, it supports any system with a Linux kernel. For Windows and MacOS, testing has not been conducted due to resource limitations.

First clone the project
```
git clone https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN
cd PhD_Code_Zhengyang_LAN
```
The dependencies of this project can be installed by:

```
pip install -r requirements.txt
```

## How to use it?

For purpose of convenience, most of the code is presented in Notebook format, with the basic code and architecture located in the utils and classifiers folders by .py file. 

This repository gives the example of datasets TDvsCPu, but the datasets and network can be changed to others too.

Before start you should download and unzip the file "paper1&2_data.zip" from the data provided. It contains the necessary training dataset.

### Deep learning-based 3DGK diagnostic tool

Just open the [Demo_Train_evaluation.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/Demo_Train_evaluation.ipynb). You will see a demo train (the training part) and evaluation the performance of dataset TDvsCPu by ResNet. You can load others datasets and change the ResNet to LSTM or InceptionTime. 
You can download “paper1_trained_model.zip” from the provided dataset, unzip it, and select the dataset you need for evaluation.


### XAI + Diagnostic Tools

You can download and unzip file "Paper2_trained_model.zip" and copy them to the repository to get the trained model. Or train it by yourself.

#### For the Feature selection, 
just run:

```
Python Feature_selection_forward_ResNet_TDvsCPu.py
```
for forward feature selection of ResNet - TDvsCPu.

And run:
```
Python Feature_selection_backward_ResNet_TDvsCPu.py
```

for backward feature selection of ResNet - TDvsCPu.

The ranking of angles will be displayed in the file name like: "_3_1_5_6_...". You can also save it to a .npy file to use it easily.
The Notebook [Feature_selection_evaluation.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/Feature_selection_evaluation.ipynb) how to get the accuracy of each step of feature selection.

#### To evaluate the LIME, Deep SHAP and Integrated gradients:

Just open the [XAI-Captum-run-save.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/XAI-Captum-run-save.ipynb). You will see how it be implemented with Captum by ResNet trained by dataset TDvsCPu. After you get the raw value of the results, the [XAI_total_relevance_Norm_TDvsCPu.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/XAI_total_relevance_Norm_TDvsCPu.ipynb) shows you how I do the normalization and save the ranking of angles.


#### To plot the results
Just open the [Plot_TDvsCPu_all_XAI.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/Plot_TDvsCPu_all_XAI.ipynb). It loads all XAI results and plot with sequential training accuracy (proved less critical angles can achieve better performance). If you just need to plot the results, you can just download "Paper 2 - XAI results.zip" file, unzip it and then copy "FS_result","XAI_ranking" and "XAI_value" of TDvsCPu to the repository to the repository to save the time.
 
### XAI evaluation (quantitative analysis)

#### Robustness
The evaluation of robustness is performed using STD. For feature selection, it is the STD of accuracy. For other XAI methods, it is the STD of XAI output values. The calculation is manual, so there is no code. The "mean" items in this [Excel](https://docs.google.com/spreadsheets/d/1c4OczUgfoRxpkj8lkq6GkfcxPxPB_mrE2O-3hxY_IvA/edit?usp=sharing) table show the process. And the Plot of the results shows in [Plot_of_quantitative_analysis_robustness.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/Plot_of_quantitative_analysis_robustness.ipynb).


#### Fidelity
The evaluation of Fidelity is performed using the $\Delta accuracy$ after remove half of important/ no important angles based on the ranking of the important angles of different XAI methods. The calculation is manual, so there is no code. The "Drop - first" and "Drop - last" items in this [Excel](https://docs.google.com/spreadsheets/d/1QGWbR75-wQdB0ambuQJdiiRhmNCxH9V6WyqNS4IUEzc/edit?usp=sharing) table show the process. And the Plot of the results shows in [Plot_of_quantitative_analysis_fidelity.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/Plot_of_quantitative_analysis_fidelity.ipynb).


## For prediction of given cycles

[Prediction_cycles.ipynb](https://github.com/LANZhengyang/PhD_Code_Zhengyang_LAN/blob/main/Prediction_cycles.ipynb) shows how to do it. You can use numpy to load any cycles with shape [n, 22, 101] to do it.

