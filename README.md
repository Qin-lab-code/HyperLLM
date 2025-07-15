# HyperLLM
The code of our paper Large Language Models Enhanced Hyperbolic Space Recommender Systems (SIGIR 2025)

This repository provides the implementation of HyperLLM based on the HGCF base model. Implementations for other base models are currently being organized and will be released soon.

The `geoopt` package used in the code can be installed via `pip install geoopt`.

You can run the following code to reproduce the results in our paper.

```
python phase1.py --dataset toys --device cuda:0 --num_experts 16 --margin 0.1
python main.py --dataset toys --device cuda:0 --emb 16_50_0.1 --margin 1.2 --margin1 0.6

python phase1.py --dataset sports --device cuda:0 --num_experts 12 --margin 0.2
python main.py --dataset sports --device cuda:0 --emb 12_50_0.2 --margin 1.8 --margin1 1.3

python phase1.py --dataset beauty --device cuda:0 --num_experts 16 --margin 0.1
python main.py --dataset beauty --device cuda:0 --emb 16_50_0.1 --margin 1.6 --margin1 0.9
```
The dataset can be accessed at the following link: [Google Drive Dataset](https://drive.google.com/drive/folders/1d-jTWhRwip5X8S9GavMpi9xbxwnI9BvH?usp=sharing)
