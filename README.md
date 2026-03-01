# **ViGSA: Multi-Task Aspect-Based Sentiment Analysis with Auxiliary Embeddings & Global Sentiment Integration for Vietnamese Restaurant Reviews**

👉 If you find this project interesting and useful, please give it a **star ⭐** to support us!

### **Authors**

[**Dien X. Tran**](https://github.com/xndien2004), Kien-Cao Van, Tinh-Nguyen Huu, Hoang-Tuan Dao-Xuan, Hung-Nguyen Viet, Khanh-Duy Cao-Phan

---

## 📌 Overview

**ViGSA** (**Vietnamese Global Sentiment Auxiliary**) is a **multi-task** model for Vietnamese **Aspect-Based Sentiment Analysis (ABSA)**.
The model jointly performs:

1. **Aspect Detection**
2. **Aspect-level Sentiment Classification**
3. **Global Sentiment Prediction**

Key innovations:

* **Auxiliary Aspect Embeddings** from natural language descriptions
* **Global Sentiment Supervision** to improve aspect-level understanding
* **Multi-loss Training** (Cross-Entropy + Asymmetric + Contrastive Loss)

**Results**: Achieves **state-of-the-art** performance on **UIT-ABSA** and **VLSP-2018** datasets, proving the effectiveness of integrating semantic priors with multitask learning in **low-resource** Vietnamese ABSA.

---

## ✨ Features

* **Multi-task Learning** → Aspect detection + Sentiment classification + Global sentiment
* **Auxiliary Semantic Embeddings** → Improves aspect representation
* **Multi-domain Support** → Trained and evaluated on **Restaurant** domain (UIT-ABSA, VLSP-2018)
* **Transformer-based Encoder** → Supports InfoXLM, PhoBERT, ViT5, RemBERT
* **Contrastive Learning** → Better representation via supervised contrastive loss
* **Attention Mechanisms** → Multi-head attention for feature extraction

---

## 📂 Project Structure

```
ViGSA/
├── main.py                 # Entry point for training/evaluation
├── requirements.txt        # Python dependencies
├── train.sh                # Training script
├── dataset/                # Dataset processing modules
├── evaluation/             # Evaluation metrics/utilities
├── model/                  # Model architecture & losses
│   ├── train.py             # Model training logic
│   └── loss.py              # Custom loss functions
└── processing/              # Data preprocessing utilities
```

---

## ⚙️ Installation

```bash
git clone https://github.com/xndien2004/ViGSA
cd ViGSA
pip install -r requirements.txt
```

**Requirements**

* Python ≥ 3.7
* PyTorch ≥ 1.10
* PyTorch Lightning ≥ 1.5
* Transformers ≥ 4.10
* CUDA-enabled GPU (recommended)

---

## 📊 Dataset

* **UIT-ABSA** (Restaurant domain)
* **VLSP-2018 ABSA** (Restaurant domain)
* Supports **multi-task** data format:

  * `Train.txt`, `Dev.txt`, `Test.txt`

---

## 🚀 Usage

### Train with default parameters

```bash
bash train.sh
```

### Train with custom parameters

```bash
python3 -m absa_project.main \
    --train_file "path/to/train.txt" \
    --val_file "path/to/dev.txt" \
    --test_file "path/to/test.txt" \
    --model_name "microsoft/infoxlm-large" \
    --topk_layer 4 \
    --batch_size 40 \
    --epochs 50 \
    --learning_rate 2e-5 \
    --output_dir "output/experiment_name" \
    --save_top_k 1 \
    --patience 10 \
    --max_length 256
```

---

## 🏗 Model Architecture

1. **Encoder** → Transformer-based (InfoXLM, PhoBERT, ViT5, RemBERT)
2. **Multi-head Attention** → Enhanced feature representation
3. **Aspect Detection** → Binary classification per aspect
4. **Aspect Sentiment Classification** → 4-class per aspect
5. **Global Sentiment Classification** → Overall sentiment prediction
6. **Pairwise Cosine Contrastive Learning** → Improves inter-class separation

**Loss Functions**:

* **Aspect Detection** → Asymmetric Loss
* **Aspect Sentiment** → Asymmetric Loss
* **Global Sentiment** → Weight Cross-Entropy Loss
* **Representation Learning** → Contrastive Loss

---

## 📈 Evaluation

Metrics:

* **Aspect Detection** → Precision, Recall, F1
* **Sentiment Classification** → Precision, Recall, F1
* **Overall** → Combined task metrics

---

## 🏆 Results

* **SOTA** performance on UIT-ABSA & VLSP-2018
* Detailed logs & checkpoints stored in `output/` directory

---

## 📚 Citation

```bibtex
@misc{dien2025vigsa,
  title={ViGSA: A Multi-Task Aspect-Based Sentiment Analysis Model with Auxiliary Embedding and Global Sentiment Integration for Vietnamese Restaurant Reviews},
  author={Tran, Dien X. and Van, Kien-Cao and Huu, Tinh-Nguyen and Dao-Xuan, Hoang-Tuan and Viet, Hung-Nguyen and Cao-Phan, Khanh-Duy},
  journal={Expert Systems with Applications (under review)},
  year={2025}
}
```
