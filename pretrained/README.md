---
license: other
license_name: seed-story-license
license_link: https://huggingface.co/TencentARC/SEED-Story/blob/main/License_Seed-Story.txt
datasets:
- TencentARC/StoryStream
language:
- en
library_name: seed-story
pipeline_tag: text-to-image
---
# SEED-Story
[![arXiv](https://img.shields.io/badge/arXiv-2407.08683-b31b1b.svg)](https://arxiv.org/abs/2407.08683)
[![Static Badge](https://img.shields.io/badge/Dataset-Huggingface-yellow)](https://huggingface.co/datasets/TencentARC/StoryStream)
[![Static Badge](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/TencentARC/SEED-Story) 

**TL;DR:** We introduce SEED-Story, a MLLM capable of generating multimodal
long stories consists of rich and coherent narrative texts, along with images that are consistent in characters and
style. We also release the StoryStream Dataset for build this model.

## Model Weights
We release the pretrained Tokenizer, the pretrained De-Tokenizer, the pre-trained foundation model **SEED-X-pretrained**, 
the StoryStream instruction-tuned MLLM **SEED-Story-George**, and the StoryStream tuned De-Tokenizer in **Detokenizer-George**

Please download the checkpoints and save them under the folder `./pretrained`.

You also need to download [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat), and save them under the folder `./pretrained`. Please use the following script to extract the weights of visual encoder in Qwen-VL-Chat.
```bash
python3 src/tools/reload_qwen_vit.py
```

## Citation
If you find the work helpful, please consider citing:
```bash
@article{yang2024seedstory,
      title={SEED-Story: Multimodal Long Story Generation with Large Language Model}, 
      author={Shuai Yang and Yuying Ge and Yang Li and Yukang Chen and Yixiao Ge and Ying Shan and Yingcong Chen},
      year={2024},
      journal={arXiv preprint arXiv:2407.08683},
      url={https://arxiv.org/abs/2407.08683}, 
}
```

## License
`SEED-Story` is licensed under the Apache License Version 2.0 except for the third-party components listed in [License](License_Seed-Story.txt).