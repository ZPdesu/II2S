# Improved StyleGAN Embedding: Where are the Good Latents? 

> **Improved StyleGAN Embedding: Where are the Good Latents? **<br/>
[Peihao Zhu](https://github.com/ZPdesu),
[Rameen Abdal](https://github.com/RameenAbdal),
[Yipeng Qin](https://scholar.google.com/citations?user=ojgWPpgAAAAJ&hl=en),
[John Femiani](https://scholar.google.com/citations?user=rS1xJIIAAAAJ&hl=en),
[Peter Wonka](http://peterwonka.net/)<br/>


> [arXiv](https://arxiv.org/abs/2012.09036) | [BibTeX](#bibtex) | [Video](https://youtu.be/6grbAFtKvBU)


> **Abstract** StyleGAN is able to produce photorealistic images that are almost indistinguishable from real photos. The reverse problem of finding an embedding for a given image poses a challenge. Embeddings that reconstruct an image well are not always robust to editing operations. In this paper, we address the problem of finding an embedding that both reconstructs images and also supports image editing tasks. First, we introduce a new normalized space to analyze the diversity and the quality of the reconstructed latent codes. This space can help answer the question of where good latent codes are located in latent space. Second, we propose an improved embedding algorithm using a novel regularization method based on our analysis. Finally, we analyze the quality of different embedding algorithms. We compare our results with the current state-of-the-art methods and achieve a better trade-off between reconstruction quality and editing quality.


## Description
Official Implementation of II2S.

**KEEP UPDATING !**

```
python main.py
```


## Updates

**18/12/2021** Add a rough version of the project.


## Todo List
* add a detailed readme
* update code
* ...


## BibTeX

```
@misc{zhu2020improved,
    title={Improved StyleGAN Embedding: Where are the Good Latents?},
    author={Peihao Zhu and Rameen Abdal and Yipeng Qin and John Femiani and Peter Wonka},
    year={2020},
    eprint={2012.09036},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
