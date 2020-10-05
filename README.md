# README

The PyTorch implementation for "Exploring Semantic Capacity of Terms" (EMNLP '20).



## Installation

```
pip install -r requirements.txt
python setup.py build_ext --inplace
```



## Run

```
python main.py -data cs
```



## Options

You can check out options using

```
python main.py --help
```



## Directory Tree

> The description of data is annotated

```
├── hype
│   ├── checkpoint.py
│   ├── common.py
│   ├── graph_dataset.pyx
│   ├── graph.py
│   ├── __init__.py
│   ├── lorentz.py
│   ├── manifold.py
│   ├── rsgd.py
│   ├── sn.py
│   ├── train.py
│   └── utils.py
├── data
│   ├── cs                         //computer science
│   │   ├── lemma_dict_5.txt       //lemmatization map
│   │   ├── npmi_3.txt             //npmi values (k=3)
│   │   ├── npmi_5.txt             //npmi values (k=5)
│   │   ├── wiki_pairs_5.txt       //hypernym-hyponym pairs
│   │   ├── wiki_pairs_sample.txt  //sampled pairs
│   │   └── wiki_term_level.txt    //term's level
│   ├── math                       //mathematics
│   │   ├── lemma_dict_5.txt
│   │   ├── npmi_3.txt
│   │   ├── npmi_5.txt
│   │   ├── wiki_pairs_5.txt
│   │   ├── wiki_pairs_sample.txt
│   │   └── wiki_term_level.txt
│   └── phy                        //physics
│       ├── lemma_dict_5.txt
│       ├── npmi_3.txt
│       ├── npmi_5.txt
│       ├── wiki_pairs_5.txt
│       ├── wiki_pairs_sample.txt
│       └── wiki_term_level.txt
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```



## Acknowledgment

We built the training framework partly based on Facebook Research's [Poincaré Embedding](https://github.com/facebookresearch/poincare-embeddings), which is licensed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).



## Citation

If you find this code useful, please kindly cite our paper:

```
@inproceedings{huang2020exploring,
  title={Exploring Semantic Capacity of Terms},
  author={Huang, Jie and Wang, Zilong and Chang, Kevin Chen-Chuan and Hwu, Wen-mei and Xiong, Jinjun},
  booktitle={Proceedings of EMNLP},
  year={2020}
}
```


