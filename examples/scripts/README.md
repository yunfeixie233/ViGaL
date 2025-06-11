<div align="center">
    <img alt="MM-Eureka logo" src="./docs/logo.png" style="height: 200px;" />
</div>


<div align="center">

# MM-EUREKA

</div>

<div align="center">
<p align="center">
  📖<a href="https://github.com/ModalMinds/MM-EUREKA/blob/qwen/MM_EUREKA_Tech_Report.pdf">Report</a> |
  📊<a href="https://huggingface.co/datasets/FanqingM/MMK12">MMK12 Datasets & Benchmark</a> |
  🤗<a href="https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B">MM-Eureka-Qwen-7B</a> |
   🤗<a href="https://huggingface.co/FanqingM/MM-Eureka-Qwen-32B">MM-Eureka-Qwen-32B</a>
</p>
</div>

<hr>
<div align="center">
<p style="text-align: center;">MM-EUREKA: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning<p>
</div>
<hr>
<div align="center">
<a href="https://github.com/ModalMinds/MM-EUREKA/blob/qwen/MM_EUREKA_Tech_Report.pdf">[[Report Link]]</a>
</div>



## 🎯Overview

We present **MM-Eureka-Qwen-7B** and **MM-Eureka-Qwen-32B**, both are powerful multimodal reasoning models that successfully extend large-scale rule-based reinforcement learning (RL) to multimodal reasoning. Compared to the previous version of MM-EUREKA based on InternVL, we have made improvements in model architecture, algorithms, and data. For instance, MM-Eureka-Qwen-7B achieves **66.1** on MMK12 evaluation sets, only 0.2 points below InternVL-2.5-78B. On MathVista(testmini), it reaches **73.0**, even surpassing InternVLVL-2.5-78B. MM-Eureka-Qwen-32B demonstrates stronger performance, scoring **72.3** on MMK12 evaluation sets, which exceeds both Qwen2.5-VL-72B's **70.3** and closed-source models like Gemini2-Flash, ranking second only to o1's **73.9**. On commonly used multimodal mathematical reasoning benchmarks, MM-Eureka-Qwen-32B achieves **73.4** on WeMath, outperforming all open-source models and most closed-source models including Claude3.7 Sonnet. On MathVista, it reaches **74.8**, surpassing all open-source and closed-source models. Both variants demonstrate significant improvements in multidisciplinary K12 and mathematical reasoning performance, outperforming most open-source models of similar sizes.

**Core Improvements:**

1. We further iterate the codebase to support algorithms including Online Filter, [ADORA](https://github.com/ShadeCloak/ADORA?tab=readme-ov-file), and [DAPO](https://arxiv.org/abs/2503.14476).
2. We open-source self-collected MMK12, which has 15k diverse and high-quality samples and 2k MCQs for Math, Physics, Chemistry, Biology for evaluation.
3. We train the MM-Eureka-Qwen-7B and MM-Eureka-Qwen-32B, which are the almost top performer in multimodal reasoning within similar size open-source models. Especially for Multidisciplinary K12 tasks.

🔥We open-source our complete pipeline to foster further research in this area. We release all our codes, models, data, etc. at [MM-EUREKA-Qwen](https://github.com/ModalMinds/MM-EUREKA/tree/qwen).

## 🗞️ News

- **[2025/04/15]** We released `MM-Eureka-Qwen-7B` , `MM-Eureka-Qwen-32B` and `MMK12`.
  - 📖 Report: [MM-Eureka-Qwen-Report](https://github.com/ModalMinds/MM-EUREKA/blob/qwen/MM_EUREKA_Tech_Report.pdf)
  - 🤗 Model: [MM-Eureka-Qwen-7B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B)
  - 🤗 Model: [MM-Eureka-Qwen-32B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-32B)
  - 📊 Dataset: [MMK12](https://huggingface.co/datasets/FanqingM/MMK12)
  - 🚀Code: [MM-Eureka-Qwen-Code](https://github.com/ModalMinds/MM-EUREKA/tree/qwen)

- **[2025/03/27]** We released `MM-Eureka-Qwen`.
  - 📖 Report: [MM-Eureka-Qwen-Report](https://jagged-court-d9d.notion.site/MM-Eureka-Qwen-1c13cc5a384880ffbd2de24e1dee052d)
  - 🤗 Model: [MM-Eureka-Qwen-7B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B)
  - 📊 Dataset: [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)
  - 🚀Code: [MM-Eureka-Qwen-Code](https://github.com/ModalMinds/MM-EUREKA/tree/qwen)

- **[2025/03/07]** We released `MM-Eureka`.
  - 📖 Paper: [MM-Eureka-paper](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf)
  - 🤗 Model: [MM-Eureka-8B](https://huggingface.co/FanqingM/MM-Eureka-8B) & [MM-Eureka-Zero-38B](https://huggingface.co/FanqingM/MM-Eureka-Zero-38B)
  - 📊 Dataset: [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)
  - 🚀Code: [MM-Eureka-Code](https://github.com/ModalMinds/MM-EUREKA/tree/internvl)

## 🚀 Features

This repository is built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), introducing several key enhancements:

- **Multimodal RFT Support**: Extends OpenRLHF to incorporate **vision-language models (VLMs)**, currently supporting **InternVL**, enabling multimodal reasoning capabilities.
  - Currently support **RLOO**, **REINFORCE++**, **GRPO** training using Ray.
  - vLLM integration and distributed training.
  - Support hybrid engine (`--colocate_all_models`, `--vllm_enable_sleep`).
- **Better Rule-based Reward support**: Better training visualization for Rule-based Rewards (i.g. Format Reward, Accuracy Reward, Repetition Penalty)
- **Enhanced Online Filtering**: Filtering out experiences based on Accuracy Reward during training as in [PRIME](https://github.com/PRIME-RL/PRIME)
  - Use `--enable_accuracy_filter`, `--freezing_filter_steps`, `--accuracy_lower_bound`, `--accuracy_upper_bound` to control the behavior of online accuracy filter.
- **ADORA**: Enable Adaptive Online Rollout Adjustment by using `--use_adora` and `--adora_lamda` as in [ADORA](https://five-stetson-b51.notion.site/Training-Reasoning-Model-with-Dynamic-Advantage-Estimation-on-Reinforcement-Learning-1a830cc0904681fa9df3e076b6557a3e).
- **DAPO**: You can use `--use_dapo` to enable DAPO loss during training as in [DAPO](https://arxiv.org/abs/2503.14476).


## 🤖 Models

Based on the key factors identified by https://github.com/ModalMinds/MM-EUREKA for achieving stable training, we enhanced the model, dataset, and algorithmic modules. Specifically, we maintained the strategy of omitting the KL divergence term and applying data filtering, while implementing the following critical modifications:

- The base model was upgraded from InternVL2.5-8B-Instruct to the more powerful Qwen2.5-VL-7B-Instruct.
- The Vision Transformer (ViT) module was frozen during training.
- The underlying RL algorithm was replaced with [GRPO](https://arxiv.org/pdf/2402.03300), instead of the previously used RLOO.
- The data filtering strategy was transitioned from an offline approach to an online approach.
- Additional data from the K12 dataset was collected, expanding the total dataset size to 15,000 samples.

| Model                  | MathVista | MathVerse | MathVision | OlympiadBench | WeMath | MMK12 |
|------------------------|-----------|-----------|------------|---------------|--------|-------|
| Claude3.7-Sonnet       | 66.8      | 52.0      | 41.3       | 48.9          | 72.6   | 55.3  |
| GPT-4o                 | 63.8      | 50.2      | 30.4       | 35.0          | 68.8   | 49.9  |
| o1                     | 73.9      | 57.0      | 60.3       | 68.0          | 98.7   | 73.9  |
| Gemini2-flash          | 70.4      | 59.3      | 41.3       | 51.0          | 71.4   | 65.2  |
| Qwen-2.5-VL-7B         | 68.2      | 47.9      | 25.4       | 20.2          | 62.1   | 53.6  |
| Qwen-2.5-VL-32B        | 74.7/71.7 | 49.9      | **40.1**       | 30.0          | 69.1   | 66.8  |
| Qwen-2.5-VL-72B        | **74.8**      | **57.6**      | 38.1       | **40.4**          | 72.4   | 70.5  |
| InternVL2.5-VL-78B     | 72.3      | 51.7      | 32.2       | 31.1          | 66.3   | 61.6  |
| QVQ-72B-Preview        | 71.4      | 48.2      | 35.9       | 33.2          | 65.4   | 61.5  |
| Adora-7B               | 73.5      | 50.1      | 23.0       | 20.1          | 64.2   | 58.1  |
| R1-Onevision-7B        | 64.1      | 47.1      | 29.9/23.5  | 17.3          | 61.8   | 39.8  |
| **MM-Eureka-Qwen-7B**  | 73.0      | 50.3      | 26.9       | 20.1          | 66.1   | 64.5  |
| **MM-Eureka-Qwen-32B** | **74.8**      | 56.5      | 34.4       | 35.9          | **73.4**   | **72.2**  |



- 🤗 [MM-Eureka-Qwen-7B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-7B)
- 🤗 [MM-Eureka-Qwen-32B](https://huggingface.co/FanqingM/MM-Eureka-Qwen-32B)

## 🏁 Getting Started

### 📦 Installation

```shell
git clone https://github.com/ModalMinds/MM-EUREKA.git
git checkout qwen
cd MM-EUREKA
pip install -e .[vllm]
pip install flash_attn --no-build-isolation
```

### 📂 Data Preparation

You can download our training data from [MMK12](https://huggingface.co/datasets/FanqingM/MMK12)

Once downloaded, refer to the section below for additional data formation.

#### Custom dataset

For custom dataset, format your data in to a JSONL file, where each entry is a dictionary organized in the following format.

```json
{
  "id": "0",
  "message": "[{\"role\": \"user\", \"content\": [{\"type\": \"image\", \"image\": \"file:///path/to/your/image.jpg\"}, {\"type\": \"text\", \"text\": \"How many cats in the image?\"}]}]",
  "answer": "gt that could be parsed and verified by math_verify"
}
```

### 🌐 Start Training

Before starting your own training, ensure that the paths in the provided training scripts are correctly set and that environment variables like `$MASTER_ADDR` and `$NODE_RANK` are properly configured.

**start MM-Eureka-Qwen-7B training**

- for single node

  ```shell
  sh examples/scripts/train_mm_eureka_qwen_7b_single_node.sh
  ```

- for multiple node

  ```shell
  sh examples/scripts/train_mm_eureka_qwen_7b_multi_node.sh
  ```

## ⭐ Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=ModalMinds/MM-EUREKA&type=Date)](https://star-history.com/#ModalMinds/MM-EUREKA&Date)

## 🤝 Contribution

MM-Eureka is still under active development, if you want to contribute, please feel free to make a pull request or create an issue.

Please refer to `CONTRIBUTING.md` before you dive in！

## 📬 Contact

If you have any questions or would like to engage with our community, feel free to scan the QR code below to join our WeChat group.

<div align="center">
<img alt="MM-Eureka logo" src="docs/wechat_qr.png" style="height: 400px;" />
</div>

## 🎓 Acknowledgements

We acknowledge the outstanding open-source contributions from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [LMM-R1](https://github.com/TideDra/lmm-r1) and [vLLM](https://github.com/vllm-project/vllm). We also extend our gratitude to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [InternVL](https://github.com/OpenGVLab/InternVL) and [QwenVL](https://github.com/QwenLM/Qwen2.5-VL) for their open-source techniques and base models, which have enabled us to further our exploration.

## 📜 Citation
```
@article{meng2025mm,
  title={MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning},
  author={Meng, Fanqing and Du, Lingxiao and Liu, Zongkai and Zhou, Zhixiang and Lu, Quanfeng and Fu, Daocheng and Shi, Botian and Wang, Wenhai and He, Junjun and Zhang, Kaipeng and others},
  journal={arXiv preprint arXiv:2503.07365},
  year={2025}
}
```

