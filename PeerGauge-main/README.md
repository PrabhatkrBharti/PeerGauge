# **PeerGauge: a Dataset for Peer Review Disagreement and Severity Gauge**

This repository contains the dataset, code, and methodologies developed for the research project, *PeerGauge: a Dataset for Peer Review Disagreement and Severity Gauge*. The study introduces a **Contradiction Severity Score**, providing a nuanced assessment of disagreements among reviewers, ranging from -5 (strong agreement) to +5 (strong contradiction). The project leverages **NLP models**, including fine-tuned LLMs, BiLSTM + Attention, and LSTM + Attention architectures, for automating the detection and scoring of contradictions in scientific peer reviews.

---

## **Folder Structure**

```plaintext
├── .gitattributes
├── LICENSE
├── README.md
├── data
│   ├── PeerGauge-dataset.csv          # Main dataset with annotated peer review pairs
│   ├── baseline_dataset.csv           # Dataset used for initial baseline comparisons
│   └── inter_annotator_agreement.csv  # Metrics for agreement between annotators
├── images
│   ├── bilstm_architecture.png        # BiLSTM + Attention architecture diagram
│   ├── bilstm_metrics.png             # Performance metrics for BiLSTM
│   ├── fine_tune_architecture.png     # Diagram for fine-tuned LLM setup
│   ├── flow.png                       # Workflow of the annotation process
│   ├── lstm_architecture.png          # LSTM + Attention architecture diagram
│   └── lstm_metrics.png               # Performance metrics for LSTM
└── scripts
    ├── bilstm-scibert.ipynb           # Implementation of BiLSTM + SciBERT
    ├── contradiction_transformer.ipynb# Transformer-based contradiction detection
    ├── fine-tune-llm.ipynb            # Fine-tuning LLMs for contradiction detection
    └── lstm_attention.ipynb           # Implementation of LSTM + Attention model
```

---

## **Project Overview**

### **Objective**
The project aims to assist editors and conference chairs in managing reviewer contradictions by:
1. Quantifying contradiction severity on a Likert scale (-5 to +5).
2. Automating contradiction detection using advanced NLP models.
3. Enhancing decision-making in high-volume peer review processes.

### **Dataset**
- **PeerGauge Dataset**: Annotated dataset derived from ICLR (2017-2020) and NeurIPS (2016-2019) review pairs.
- **Contradiction Severity Scale**:
  - **+5 (Strong Contradiction)**: Complete opposition between reviews.
  - **0 (Neutral)**: No clear agreement or contradiction.
  - **-5 (Strong Agreement)**: Complete alignment between reviews.
- **Statistics**:
  - **28k Review Pairs**
  - **50k Comments**
  - **8 Key Aspects**: Clarity, originality, soundness, replicability, etc.

### **Key Features**
- **Models Implemented**:
  - **Fine-Tuned LLM**: Mistral-7B-Instruct with LoRA-based parameter tuning.
  - **BiLSTM + SciBERT**: For domain-specific embeddings and contradiction detection.
  - **LSTM + Attention**: Enhanced feature extraction for nuanced contradictions.
  - **HF Transformers**: Leveraging RoBERTa for state-of-the-art performance.
- **Annotation Methodology**:
  - Human and active learning-based hybrid annotation process.
  - Likert scale severity scoring ensures detailed analysis of contradictions.

---

## **Results and Analysis**

### **Main Results**
- **Best Performance**: Fine-tuned LLM (Mistral-7B-Instruct) achieved the highest accuracy and F1-score.
- **Qualitative Insights**:
  - Severe contradictions were common in critical aspects like soundness and substance.
  - Minor agreements appeared in aspects like clarity and replicability.

### **Performance Metrics**

| Model                  | Precision | Recall | F1-Score | Accuracy |
|------------------------|-----------|--------|----------|----------|
| Fine-Tuned LLM         | 90.91     | 92.04  | 91.47    | 91.41    |
| BiLSTM + SciBERT       | 83.73     | 90.75  | 87.10    | 86.62    |
| LSTM + Attention       | 76.15     | 88.00  | 81.65    | 80.30    |

---

### **Scripts**
- Run Jupyter notebooks from the `scripts/` folder for model training and evaluation.
- Use the `PeerGauge-dataset.csv` for experiments or as input to train custom models.

---

## **Contact**

For queries, please reach out:
- **Prabhat Kumar Bharti**: [dept.csprabhat@gmail.com](mailto:dept.csprabhat@gmail.com)
- **Mihir Panchal**: [mihirpanchal5400@gmail.com](mailto:mihirpanchal5400@gmail.com)
- **Viral Dalal**: [viraldalal04@gmail.com](mailto:viraldalal04@gmail.com)
- **Mayank Agarwal**: [mayank265@iitp.ac.in](mailto:mayank265@iitp.ac.in)
- **Asif Ekbal**: [asif@iitp.ac.in](mailto:asif@iitp.ac.in)

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
