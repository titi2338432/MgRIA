# Multi-granularity repurchase interval-aware recommendation with small item-sets

Understanding users' repurchase patterns is crucial for improving the quality of item recommendations. Many studies have straightforwardly modeled the intervals between users' repeat purchases using a single distribution. However, the empirical distribution of repurchase intervals for certain products on e-commerce platforms often resembles a mixture distribution. Typically, this mixture consists of one major unimodal distribution coupled with multiple Gaussian distributions of different time granularities. Currently available recommendation systems cannot achieve optimal effectiveness if this fact is not accounted for, especially when recommending with a small item-set. Based on this finding, we propose MgRIA, a BERT-based recommendation model with a user repeat purchase feature block. This module includes a flexible scoring mechanism designed to accommodate data following a mixture distribution, as observed in repeat purchase behaviors. We applied MgRIA and several existing recommendation methods to various datasets and found that MgRIA achieves the best overall performance.

recommendation system; repurchase time interval; multi-granularity; mixture distribution; attention mechanism

codes are to be released soon

Files and Folders

- **datasets** the datasets to train and test. Please note that due to limited space, the real large datasets should be downloaded from certain websites, as the `datasets.ipynb` instructs the way of processing raw datasets.

- **methods** includes baseline algorithms and ours. Please note that `IIBidder_agent` is our algorithm. DRLB, Uniform, Normal, Lin, and Gamma are included in the compare experiment. AC_GAIL_agent, PPO_GAIL_agent, and PPO_agent are included in the ablation experiment.

- **README.md** this file

- **ablation_exp.py** to do ablation study

- **compared_exp.py** to do comparison among different baselines (DRLB, Uniform, Normal, Lin, and Gamma)

- **globals.py** global variables

- **main.py** main entrance of the experiments. to envoke `run.py`, `post.py`, and `plot.py`

- **plot.py** read `final.csv`, plot the figures as .pdf used for the paper to store in `figures` folder, and generate latex tables as .tex files for the papers.

- **post.py** post-process, to put together all the intermediate results in to one `final.csv` file.

- **preprocess.py** read data from 'datasets' folder, and preprocess for compare experiment and ablation experiment.

- **requirements.txt** for install the conda virtual env.

- **rtb_environment.py** create a bidding environment for agent

## Usage

1. Install Python 3.9. For convenience, execute the following command.

```shell
pip install -r requirements.txt
```

Another more elegant way to reproduce the result, of course, is use conda virtual environment, as widely appreciated. Typically by entering the following command before the above pip installation:

```shell
conda create -n ImagineRTB python=3.9
```

We are not going to discuss the details here.

2. Prepare Data.

Download the original datasets from [IPINYOU](https://contest.ipinyou.com/) and [YOYI](https://apex.sjtu.edu.cn/datasets/7). Process the raw dataset according to the instruct in `datasets.ipynb` in the datasets folder. Considering the dataset size and space limitations, we provide partial sample datasets for users to directly invoke and experiment with.

3. Train and evaluate model. You can adjust parameters in global.py and reproduce the experiment results as the following examples:

```python
python3 main.py
```

As a scheduler, `main.py` will envoke `run.py`, `post.py`, and `plot.py` one by one, the functions of which are introduced in the **files & folders** part.

4. Check the results
- results are in .csv format at `results` folder, which are later combined together to a `final.csv` for plotting purpose.
- figures are at `figures` folder
- latex tables are at `tables` folder

## Citation

If you find this repo useful, please cite our paper.

```
@inproceedings{Li2024maria,
  title={Multi-gradularity repurchae interval aware recommendation},
  author={Yin Tang},
  journal={},
  year={2024},
}
```

## Contact

If you have any questions or suggestions, feel free to contact:

- Yin Tang <[ytang@jnu.edu.cn](mailto:ytang@jnu.edu.cn)>

Or describe it in Issues.

## Acknowledgement

This work is supported by National Natural Science Foundation of China (62272198) and by Guangdong Provincial Science and Technology Plan Project (No.2021B1111600001).
