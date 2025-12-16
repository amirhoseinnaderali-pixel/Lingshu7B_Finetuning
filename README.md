# 🩻 RadVision-7B: Intelligent Radiology Report Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-%F0%9F%94%A5-red)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Lingshu--7B-Medical-green)](https://huggingface.co/)

**RadVision-7B** is a specialized **Vision-Language Model (VLM)** for **Automated Radiology Report Generation** from Chest X-ray images. Built upon the **Lingshu-7B** medical large language model, RadVision-7B is fine-tuned on the **IU X-ray (Open-I)** dataset to bridge the gap between medical imaging and clinically grounded textual reasoning.

The system automatically generates structured radiology reports, focusing on the **Findings** and **Impression** sections commonly used in real-world clinical workflows.

---

## 🧩 System Overview

```text
┌──────────────────────────┐
│   Chest X-Ray Image      │
│     (Frontal View)       │
│      512 × 512           │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│     Vision Encoder       │
│  (X-ray Feature Extract) │
│  • Stage 1 Fine-tuned    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Multimodal Projector   │
│  (Visual → Token Space) │
│  • Stage 2 Alignment     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        LLM (7B)          │
│     Lingshu-7B Base      │
│  • Stage 3 Fine-tuned    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Radiology Report Text   │
│                          │
│  Findings:               │
│  - Pulmonary opacity...  │
│  - Cardiac silhouette... │
│                          │
│  Impression:             │
│  - Possible pneumonia    │
└──────────────────────────┘
```

---

## 🌟 Key Features

### 🧠 Advanced 3-Stage Training Pipeline

To ensure robust alignment between visual representations and medical language, training is divided into three focused stages:

1. **Stage 1 – Vision Encoder Adaptation**
   Fine-tunes the vision backbone on Chest X-ray imagery to capture domain-specific anatomical and pathological features.

2. **Stage 2 – Multimodal Projector Alignment**
   Trains the projection layer that maps visual embeddings into the LLM token space, enabling effective cross-modal fusion.

3. **Stage 3 – Language Model Fine-tuning**
   Fine-tunes the Lingshu-7B language model to generate clinically coherent, structured radiology reports.

---

### 🚀 Memory-Optimized Training

RadVision-7B is designed to be trainable on **consumer-grade GPUs**:

* **QLoRA (4-bit Quantization):** Enables efficient fine-tuning of 7B parameters with minimal memory usage.
* **Paged AdamW (8-bit):** Reduces optimizer memory footprint.
* **Gradient Checkpointing:** Trades compute for memory, allowing larger image sizes and batch configurations.

---

### 💎 Qualitative Inference Visualization

The project includes a **Medical Dashboard** for qualitative evaluation:

* Side-by-side display of:

  * Input Chest X-ray
  * AI-generated report
  * Ground-truth radiologist report
* Clean card-based layout for easy visual comparison
* Designed for debugging, demos, and research presentations

---

## 🧬 Model Details

* **Base LLM:** Lingshu-7B (Medical-domain LLM)
* **Task:** Vision-to-Text Generation
* **Input:** Frontal Chest X-ray image (512×512)
* **Output:** Structured radiology report

  * *Findings*
  * *Impression*
* **Language:** English

---

## 🛠️ Tech Stack

* **Core:** PyTorch, HuggingFace Transformers
* **Optimization:** PEFT, BitsAndBytes, QLoRA
* **Data Handling:** HuggingFace Datasets, PIL
* **Visualization:** Matplotlib, IPython.display

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/RadVision-7B.git
cd RadVision-7B
```

### 2. Install dependencies

```bash
pip install transformers>=4.49.0 accelerate bitsandbytes peft \
            qwen-vl-utils datasets pillow torchvision tqdm \
            scikit-learn matplotlib
```

---

## 🚀 Usage

All training and inference steps are contained in the Jupyter Notebook:

```
radvision-7b-iu-visionfocused-v1.ipynb
```

### Running the pipeline

```bash
jupyter notebook radvision-7b-iu-visionfocused-v1.ipynb
```

The notebook covers:

* Automatic dataset loading from Hugging Face
* 3-stage fine-tuning pipeline
* Inference and visualization on test samples

### Example Inference

```python
# Generate a visual comparison between prediction and ground truth
predict_beautiful(index=5)
```

---

## 📊 Dataset & Preprocessing

* **Dataset:** Open-I (IU Chest X-ray)
* **Source:** HuggingFace `ykumards/open-i`

### Preprocessing Steps

* Removal of samples without frontal views
* Extraction of *Findings* and *Impression* sections
* Image resizing to **512×512** using LANCZOS resampling

---

## 📈 Results (Qualitative)

Qualitative evaluation shows that RadVision-7B:

* Generates anatomically grounded descriptions
* Maintains clinically appropriate language
* Produces well-structured and coherent impressions

This project emphasizes **interpretability and report quality** over pure metric optimization.

---

## ⚠️ Medical Disclaimer

This project is intended **for research and educational purposes only**.

It is **NOT** a medical device and must **NOT** be used for clinical diagnosis, treatment, or decision-making. All outputs should be reviewed by a qualified radiologist.

---

## 🤝 Contributing

Contributions are welcome. Feel free to:

* Open an issue for bugs or feature requests
* Submit a pull request for improvements
* Suggest extensions to other imaging modalities

---

## 📝 License

This project is released under the **MIT License**. See the `LICENSE` file for details.
