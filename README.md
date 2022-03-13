# PhoneDepth
PhoneDepth: Dataset toolbox and documentation for the paper submitted to IJCAI 2022.


## Development setup
This repository is based on the setup described by the `environment.yml` file and is supposed to be used with conda. One can create the conda environment with the command: 

```
conda env create -f environment.yml
```

Note that the key dependencies are:
- Tensorflow >=2.4.1
- Tensorflow-addons >= 0.14.0 (Corresponding to Tensorflow version)
- Tensorflow-probability >=0.7 (Corresponding to Tensorflow version)


## Dataset download
We recommend you keep all the datasets under the same directory (`*/data`), although it is not necesary.

### Data-set lists
The data lists corresponding representing the splits used to acquire this paper's results can be downloaded from [here](https://drive.google.com/file/d/1uDzpz-pVIPAnabLigE1ThwF40QbhfI6-/view?usp=sharing).  It is composes of the following directory structure whear each superficial directory corresponds to each dataset:

```bash
├── MAI
│   ├── train.npy
│   └── val.npy
├── MegaDepth
│   ├── final_list
│   │   └── **
│   └── wo_ordinal_list
│       └── **
└── PhoneDepth
    ├── test_list.json
    ├── train_list.json
    └── validation_list.json
```
### Megadepth
Download and extract [MegaDepth V1](https://www.cs.cornell.edu/projects/megadepth/) dataset. Place the `MegaDepth` data-set list directory contents inside the top level directory of the dataset after extracting. 
### Mobile AI (MAI)
Download Publicly available data on: [Mobile AI](https://competitions.codalab.org/competitions/28122). You will need to be signed-in to access the data. Extrace the dataset from the compressed file. Then you should include the data-lists corresponding to `MAI` in the previous section. You want to have a file structure as follows:

```bash
MAI2021_depth_dir (as downloaded or you name it)
└── train
    ├── rgb
    ├── depth
    ├── train.npy
    └── val.npy
```

### PhoneDepth
Download dataset available in [PhoneDepth](placeholder). Extract it and place the `PhoneDepth` data-lists in the root directory of the dataset. You want to have a file structure as follows:

```bash
PhoneDepth (as downloaded or you name it)
└── train
    ├── hua
    ├── networks
    ├── p_depth_logs
    ├── p_eval_outputs
    ├── p_networks
    ├── test_list.json
    ├── train_lists.json
    └── validation_list.json
```

## Content Description
### Training
### Evaluation