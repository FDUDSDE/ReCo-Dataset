## ReCo-Dataset
This is the offical repository for **Re**sidential **Co**mmunity Layout Planning (**ReCo**) Dataset

🔥🔥🔥Our paper has been accepted by [ACMMM'23 (Oral)](https://dl.acm.org/doi/10.1145/3581783.3612465).

Detailed information for our dataset please see our article preprint at [arXiv](https://arxiv.org/abs/2206.04678)

Our dataset is under the [CC BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

ReCo dataset is designed for Community Layout Planning tasks which is one of the three typical tasks of layout planning.

![image](https://github.com/FDUDSDE/ReCo-Dataset/blob/main/images/tasks.png)
Three typical tasks of layout planning from fine- to coarse-grained (the projection relationship in the figure is for illustration only).

### Generating 2D image data from ReCo Dataset
1. Please download the dataset from https://www.kaggle.com/fdudsde/reco-dataset and put the JSON file under the main directory.
2. Please make sure that the JSON file is named as ReCo_json.json
3. You can plot one of example of the data by using "_id" as index through plot_2d_from_json.py.
4. make_image_data.py can help you to build an image dataset from ReCo_json.json file.

### Example of generated 2D image
![image](https://github.com/FDUDSDE/ReCo-Dataset/blob/main/images/data_example.png)

### Experiments
We redeveloped the code at https://github.com/eriklindernoren/PyTorch-GAN and used DCGAN as the backbone networks.

We used an Nvidia Tesla V100 to train the model for 2k epochs with a batch size of 128 per sub-experiment.

The training details of our demonstrated experiments are shown in `experiments` codes. \
We used the same hyperparameters in our four sub-experiments.

#### Model architecture
![image](https://github.com/FDUDSDE/ReCo-Dataset/blob/main/images/model.png)

## DOI
### Paper
10.1145/3581783.3612465
### Dataset
10.34740/kaggle/dsv/3689702

## Citation
### Article Citation
🔥ACMMM'23

@inproceedings{10.1145/3581783.3612465,\
author = {Chen, Xi and Xiong, Yun and Wang, Siqi and Wang, Haofen and Sheng, Tao and Zhang, Yao and Ye, Yu},\
title = {ReCo: A Dataset for Residential Community Layout Planning},\
year = {2023},\
isbn = {9798400701085},\
publisher = {Association for Computing Machinery},\
address = {New York, NY, USA},\
url = {https://doi.org/10.1145/3581783.3612465},\
doi = {10.1145/3581783.3612465},\
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},\
pages = {397–405},\
numpages = {9},\
keywords = {residential community layout, layout planning and design, dataset, layout generation},\
location = {Ottawa ON, Canada},\
series = {MM '23}\
}

arXiv

@article{reco2022article,\
  title={ReCo: A Dataset for Residential Community Layout Planning},\
  author={Chen, Xi and Xiong, Yun and Wang, Siqi and Wang, Haofen and Sheng, Tao and Zhang, Yao and Ye, Yu},\
  journal={arXiv preprint arXiv:2206.04678},\
  year={2022},\
  copyright={Creative Commons Attribution Non Commercial Share Alike 4.0 International}\
}

### Datset Citation
@misc{reco2022dataset, \
  title={ReCo:Residential Community Layout Planning Dataset}, \
  url={https://www.kaggle.com/dsv/3689702}, \
  DOI={10.34740/KAGGLE/DSV/3689702}, \
  publisher={Kaggle}, \
  author={Xi Chen and Yun Xiong and Siqi Wang and Haofen Wang and Tao Sheng and Yao Zhang and Yu Ye}, \
  year={2022}, \
  copyright={Creative Commons Attribution Non Commercial Share Alike 4.0 International}\
}

## Acknowledgments
This work is funded in part by the National Natural Science Foundation of China Projects No. U1936213, No.62176185. This work is also partially supported by the Shanghai Science and Technology Development Fund No. 19DZ1200802, and by the Shanghai Municipal Science and Technology Major Project (2021SHZDZX0100) and the Fundamental Research Funds for the Central Universities.
