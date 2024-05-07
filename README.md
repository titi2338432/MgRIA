# Multi-granularity repurchase interval-aware recommendation with small item-sets

Understanding users’ repurchase patterns is crucial for improving the quality of item recommendations on e-commerce platforms. In this study, we discovered that the empirical distribution of repurchase intervals often resembles a mixture distribution comprising a major unimodal distribution and multiple minor Gaussian distributions spanning various
time granularities. This prevents current recommendation systems, which typically model the repurchasing intervals by a unimodal distribution, from achieving the optimal effectiveness, especially when recommending a small item-set. To addessing this challenge, we propose Multi-granularity Repurchase Interval-Aware (MgRIA), a BERT-based recommendation model with a user repeat purchase feature layer. This module includes a flexible scoring mechanism designed to accommodate data following a mixture distribution, as observed in repeat purchase behaviors. In the context of the Next Basket Repurchase Recommendation problem, MgRIA has outperformed the existing matrix factorization-based and session-based recommendation models for several e-commerce datasets.

recommendation system; repurchase time interval; multi-granularity; mixture distribution; attention mechanism

codes are to be released soon

Files and Folders

- **datasets** the datasets to train and test. Please note that due to limited space, the real large datasets should be downloaded from certain websites, as the `datasets.ipynb` instructs the way of processing raw datasets.

- **methods** includes baseline algorithms and ours. 

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
conda create -n mgria python=3.9
```

We are not going to discuss the details here.

2. Prepare Data.

Download the original datasets from websites indicated in paper. Process the raw dataset according to the instruct in `datasets.ipynb` in the datasets folder. Considering the dataset size and space limitations, we provide partial sample datasets for users to directly invoke and experiment with.

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
@article{Li2024maria,
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
