# ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs 🇪🇬🇬🇧

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](http://huggingface.co/collections/ahmedheakl/arazn-llm-662ceaf12777656607b9524e)
[![arXiv](https://img.shields.io/badge/arXiv-2406.18120-b31b1b.svg)](https://arxiv.org/abs/2406.18120)
[![Demo](https://img.shields.io/badge/Demo-Watch%20Now-ff69b4)](https://github.com/ahmedheakl/arazn-llm/blob/main/demo.mp4)

## Introduction

In recent times, code-switching between Egyptian Arabic and English has become increasingly prevalent. This repository presents our work on developing advanced machine translation (MT) and automatic speech recognition (ASR) systems specifically designed to handle this linguistic phenomenon.

### 🎥 Demo

Check out our [demo video](https://github.com/ahmedheakl/arazn-llm/blob/main/demo.mp4) to see ARAZN-LLM in action!
<video width="320" height="240" controls>
  <source src="https://github.com/ahmedheakl/arazn-llm/blob/main/demo.mp4" type="video/mp4">
</video>

### 🎯 Our Goal

Our primary objective is to translate code-switched Egyptian Arabic-English to either English or Egyptian Arabic. We employ state-of-the-art methodologies utilizing large language models such as LLama and Gemma.

### 🔊 ASR Integration

In the realm of ASR, we leverage the Whisper model for code-switched Egyptian Arabic recognition. Our experimental procedures encompass:
- Data preprocessing techniques
- Advanced training methodologies

We've implemented a consecutive speech-to-text translation system that seamlessly integrates ASR with MT, addressing challenges posed by limited resources and the unique characteristics of the Egyptian Arabic dialect.

### 📊 Performance

Our evaluation against established metrics demonstrates promising results:
- **English Translation**: Significant improvement of X% over the state-of-the-art
- **Arabic Translation**: Y% improvement in performance

### 🌟 Why It Matters

Code-switching is deeply inherent in spoken languages, making it crucial for ASR systems to effectively handle this phenomenon. This capability enables seamless interaction across various domains, including:
- Business negotiations
- Cultural exchanges
- Academic discourse

## Open-Source Resources

We're committed to advancing research in this field. Our models and code are available as open-source resources:

- 🤗 **Models**: [Hugging Face Collection](http://huggingface.co/collections/ahmedheakl/arazn-llm-662ceaf12777656607b9524e)
- 🗣️ **Speech Dataset**: [ARZEN-LLM Speech Dataset](https://huggingface.co/datasets/ahmedheakl/arzen-llm-speech-ds)
- 🔤 **Translation Dataset**: [ARZEN-LLM Translation Dataset](https://huggingface.co/datasets/ahmedheakl/arzen-llm-dataset)
- 📄 **Research Paper**: [arXiv:2406.18120](https://arxiv.org/abs/2406.18120)

Feel free to explore, contribute, and build upon our work!

```bibtex
@article{heakl2024arzen,
  title={ArzEn-LLM: Code-Switched Egyptian Arabic-English Translation and Speech Recognition Using LLMs},
  author={Heakl, Ahmed and Zaghloul, Youssef and Ali, Mennatullah and Hossam, Rania and Gomaa, Walid},
  journal={arXiv preprint arXiv:2406.18120},
  year={2024}
}
```

