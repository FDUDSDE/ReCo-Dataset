## ReCo-Dataset
Our dataset is under the [CC BY-NC-SA license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### Generating 2D image data
1. Please download the data from https://www.kaggle.com/fdudsde/reco-dataset and put the JSON file under this directory.
2. Please make sure that the JSON file is named as ReCo_json.json
3. You can plot one of example of the data by using "_id" as index through plot_2d_from_json.py.
4. make_image_data.py can help you to build an image dataset from ReCo_json.json file.

### Experiments
We redeveloped the code at https://github.com/eriklindernoren/PyTorch-GAN and used DCGAN as the backbone networks.

We used an Nvidia Tesla V100 to train the model and run 5k epochs.

The training details of our demonstrated experiments are shown in `experiments` codes. 
We used the same hyperparameters in our two sub-experiments.

