
<div align="center">

#  🎲 *Play to Reason*:<br>  Learning Math through Visual Games


<p>
    <img src="fig/teaser.png" alt="ViGaL" width="300" height="auto">
</p>



<a href="https://arxiv.org/abs/2406.16860" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-ViGal-red?logo=arxiv" height="25" />
</a>
<a href="https://yunfeixie233.github.io/ViGaL/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-yunfeixie233.github.io/ViGaL/-blue.svg" height="25" />
</a>
<br>
<a href="" target="_blank">
    <img alt="HF Model: ViGaL" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-ViGaL-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="" target="_blank">
    <img alt="HF Dataset: Snake & Rotation" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-Snake%20%26%20Rotation-ffc107?color=ffc107&logoColor=white" height="25" />
</a>



<div style="font-family: charter;">
    <a href="https://yunfeixie233.github.io/" target="_blank">Yunfei Xie</a>,
    <a href="https://openreview.net/profile?id=~Yinsong_Ma1" target="_blank">Yinsong Ma</a>,
    <a href="https://voidrank.github.io/" target="_blank">Shiyi Lan</a>,
    <br>
    <a href="https://www.cs.jhu.edu/~ayuille/" target="_blank">Alan Yuille</a>,
    <a href="https://lambert-x.github.io/" target="_blank">Junfei Xiao†</a>,
    <a href="https://weichen582.github.io/" target="_blank">Chen Wei§</a>
</div>

§Corresponding Author, †Project Lead

</div>
<br>




## 🎯 Overview

We propose a novel post-training paradigm, **Visual Game Learning (ViGaL)**, where MLLMs develop out-of-domain generalization of multimodal reasoning through playing arcade-like games. Specifically, we show that post-training a 7B-parameter MLLM via reinforcement learning (RL) on simple arcade-like games like Snake and Rotation puzzle significantly enhances its downstream performance on multimodal reasoning benchmarks such as MathVista, MathVerse, and MathVision, **without seeing any worked solutions, equations, or diagrams during RL**. Remarkably, the resulting model surpasses large-scale proprietary models and models tuned directly on visual math datasets. Ablation studies indicate that distinct games unlock complementary reasoning skills, leading to improved generalization when combined. Our findings suggest a new post-training paradigm: synthetic, rule-based games can serve as controllable and scalable pre-text tasks that effectively unlock generalizable multimodal reasoning abilities in MLLMs.

## 🗞️ News

## 📋 Contents
- [Installation](#installation)
- [ViGaL Weights](#vigal-weights)
- [Data Preparation](#data-preparation)
- [Train](#train)
- [Evaluation](#evaluation)
- [Demo](#demo)

## 📦 Installation
```shell
git clone https://github.com/yunfeixie233/ViGaL.git
cd ViGaL
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

## 🤖 ViGaL Weights

## 📂 Data Preparation
You can download our training data from [ViGaL training data]()

## 🌐 Train

- For Snake game:

  ```shell
  sh examples/scripts/train_snake.sh
  ```

- For Rotation game:

  ```shell
  sh examples/scripts/train_rotation.sh
  ```
- For Snake and Rotation games:

  ```shell
  sh examples/scripts/train_snake_rotation.sh
  ```


## 🔭 Evaluation

- For MathVista, MathVision, and MathVerse:
    We use the evaluation code in the `eval/` directory. 
- For CLEVR+ and Geometry:
    Please implement the evaluation following [Reason-RFT](https://github.com/tanhuajie/Reason-RFT?tab=readme-ov-file#--evaluation)

- For other general visual evaluation:
    Please implement the evaluation following [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

## 🎮 Results

### Zero-shot generalization from gameplay to math reasoning.

| Model | Avg. | Math | | | | CLEVR+ | | | Geometry | | |
|-------|------|------|-----------|-----------|------------|---------|---------|---------|---------------|---------|--------|
| | | **Avg.** | **MathVista** | **MathVerse** | **MathVision** | **Avg.** | **CLEVR-M** | **S-CLEVR** | **Avg.** | **GeoMath** | **Geo3K** |
| **Proprietary Model** | | | | | | | | | | | |
| GPT-4o | 55.2 | 48.1 | 61.4 | 50.2 | 30.4 | 51.2 | 68.1 | 34.3 | 46.8 | 50.2 | 43.5 |
| Gemini-2.0-Flash | 56.9 | 56.4 | 73.4 | 54.6 | 41.3 | 46.3 | 64.9 | 27.6 | 54.4 | 55.3 | 53.5 |
| **General Multimodal Language Model** | | | | | | | | | | | |
| InternVL2.5-8B | 51.3 | 41.2 | 64.4 | 39.5 | 19.7 | 64.4 | 93.5 | 35.3 | 55.2 | 63.0 | 47.3 |
| Llava-OV-7B | - | - | 63.2 | 26.2 | - | 49.4 | 69.7 | 29.1 | 60.7 | 77.6 | 43.7 |
| Qwen2.5-VL-7B | 50.0 | 47.7 | 68.0 | 49.0 | 26.0 | 54.9 | 74.6 | 35.2 | 44.8 | 44.0 | 45.6 |
| **Multimodal Reasoning Model Post-Trained on Qwen2.5-VL-7B** | | | | | | | | | | | |
| R1-Onevision-7B | 46.3 | *46.8* | *64.1* | *46.4* | *29.9* | *65.1* | *75.5* | *54.7* | 35.0 | 45.4 | 24.5 |
| R1-VL-7B | 47.3 | *42.7* | *63.5* | *40.0* | *24.7* | *68.0* | *87.4* | *48.6* | *39.0* | *42.0* | *36.1* |
| MM-Eureka-Qwen-7B | 54.5 | *50.1* | *73.0* | *50.3* | *26.9* | **79.3** | 98.4 | 60.1 | *28.4* | *53.1* | *3.8* |
| Reason-RFT-Zero-7B | 50.1 | 38.1 | 60.7 | 35.3 | 18.3 | *76.2* | *99.4* | *53.0* | *54.9* | *55.0* | *54.8* |
| VLAA-Thinker-7B | 55.0 | *48.7* | *68.0* | *51.7* | *26.4* | *83.4* | *94.7* | *72.1* | *53.9* | *51.1* | *56.6* |
| OpenVLThinker-7B | 55.8 | *47.8* | *70.2* | *47.9* | *25.3* | *82.4* | *93.8* | *71.0* | *56.4* | *49.2* | *63.5* |
| Snake | - | 49.4 | 70.9 | 49.7 | 27.5 | **82.8** | 92.7 | 72.8 | 52.1 | 47.5 | 56.8 |
| Rotation | - | 49.6 | 71.0 | 51.0 | 27.3 | 81.0 | 91.7 | 70.2 | 56.1 | 51.3 | 60.9 |
| Snake + Rotation | **63.7** | **50.6** | 71.9 | 52.4 | 27.5 | 81.7 | 91.9 | 71.4 | **57.1** | 51.0 | 63.3 |

## 📜 Citation

If you find ViGaL useful for your research and applications, please cite using this BibTeX:
```bibtex

```

## 🎓 Acknowledgement

- [MM-EUREKA](https://github.com/ModalMinds/MM-EUREKA/tree/qwen): We start from the codebase from MM-EUREKA
`