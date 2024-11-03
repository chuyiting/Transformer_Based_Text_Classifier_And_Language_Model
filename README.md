# An encoder text classifier, a decoder language model following [1] Attention is All You Need

This project includes two parts. Part one contains a transformer based **Encoder text classifier** It is trained to classify speeches from **Barack Obama**, **George W. Bush**, and **George H Bush**. Part two contains a **Pure Decoder Language Model**. The efficacy of the network is shown by training on a small speech Dataset. Three architectual changes were explored in this project, namely default positonal embedding[1], linear biases[3], and disentangled attention [2]

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Reference](#references)

## Installation

1. **Download the code:**
   ```bash
   git clone https://github.com/chuyiting/Transformer_Based_Text_Classifier_And_Language_Model.git
   ```
2. **create virtual environment and install dependencies**

   ```bash
   conda env create -f environment.yml
   conda activate transformer
   ```

## Usage

**Run the project**

```bash
python main.py --model CLS
```

```bash
python main.py --model LM
```

```bash
python main.py --model ALiBi
```

```bash
python main.py --model DeBERTa
```

- For classification task, use **CLS**. Three Language Model architecture provided - **LM**, **ALiBi**, **DeBERTa**
- in the analysis folder, you can find heat maps for all the attention heads for all four models, as well as the perplexity, loss information for each of the language model.
- You are welcome to go to the `visualize.ipynb` to generate more visuals

## References

[1]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2023). Attention Is All You Need. In _Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS 2017)_. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

[2]: He, P., Liu, X., Gao, J., & Chen, W. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL 2021)_. [https://arxiv.org/abs/2006.03654](https://arxiv.org/abs/2006.03654).

[3]: Press, O., Smith, N. A., & Lewis, M. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL 2022)_. [https://arxiv.org/abs/2108.12409](https://arxiv.org/abs/2108.12409).
