# ___***MoMA: Multimodal LLM Adapter for Fast Personalized Image Generation***___

<a href='https://moma-adapter.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a> 
<a href='https://arxiv.org/abs/2404.05674'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 
<a href='https://huggingface.co/KunpengSong/MoMA_llava_7b/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue'></a>
---

**This page has been migrated to [ByteDance official](https://github.com/bytedance/MoMA/tree/main) GitHub repository.**

**Updates will be posted on the new page. **


## Introduction

we present MoMA: an open-vocabulary, training-free personalized image model that boasts flexible zero-shot capabilities. As foundational text-to-image models rapidly evolve, the demand for robust image-to-image translation grows. Addressing this need, MoMA specializes in subject-driven personalized image generation. Utilizing an open-source, Multimodal Large Language Model (MLLM), we train MoMA to serve a dual role as both a feature extractor and a generator. This approach effectively synergizes reference image and text prompt information to produce valuable image features, facilitating an image diffusion model. To better leverage the generated features, we further introduce a novel self-attention shortcut method that efficiently transfers image features to an image diffusion model, improving the resemblance of the target object in generated images. Remarkably, as a tuning-free plug-and-play module, our model requires only a single reference image and outperforms existing methods in generating images with high detail fidelity, enhanced identity-preservation, and prompt faithfulness. We commit to making our work open-source, thereby providing universal access to these advancements.

