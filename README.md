## Gimli

<p align="center">
  <img src="assets/gimli-llama.png" width="500" height="300" alt="Gimli LOTR">
</p>

Have you ever wanted to inference a small LLM in pure C? No? Well, how about pretrain a similar model completely from scratch? Still, no? Oh well, I guess I'll just leave this here then.

Train the Llama 2 LLM architecture in PyTorch then inference it with one simple 700-line C file ([run.c](run.c)). You might think that you need many billion parameter LLMs to do anything useful, but in fact very small LLMs can have surprisingly strong performance if you make the domain narrow enough. This repo is a "fullstack" train + inference solution for Llama 2 LLM architecture, with focus on minimalism and simplicity. (CREDIT: KARPATHY)

This began as a small toy fork of [llama2.c](https://github.com/karpathy/llama2.c), as a `GPU POOR` I had no business messing with large models on my machine, but still had an interesting in experimenting with different training regimes. The solution? Small models, trained on small datasets, with small batch sizes. This fork is a collection of scripts and tools to train and inference small LLMs, with a focus on simplicity.. My goal here is to be able to quickly iterate on experiments on models that show some usefulness and have some fun. 

## training

```bash
python train.py # OPTIONAL: --config=examples/20M.yml
```