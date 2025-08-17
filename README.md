# 🐱🐶 Cats vs Dogs Classification with Vision Transformer (ViT)

This repository contains a Jupyter Notebook implementation of an image classification model that distinguishes between **cats** and **dogs** using a **Vision Transformer (ViT)**.

The project demonstrates how to fine-tune a pretrained ViT model on the [Kaggle Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats), covering the full workflow from dataset preparation to evaluation.

---

## 🚀 Features

* Load and preprocess the **Cats vs Dogs** dataset
* Split data into **training**, **validation**, and **test** sets
* Apply **image augmentation** for better generalization
* Fine-tune a **pretrained Vision Transformer (ViT)** model
* Track **training loss/accuracy** and **validation performance**
* Evaluate the model on a **test dataset** with metrics

---

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/your-username/cats-vs-dogs-vit.git
cd cats-vs-dogs-vit
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```


## 📦 Requirements

Example dependencies (adjust based on your notebook):

```
torch
torchvision
transformers
tqdm
matplotlib
numpy
pandas
scikit-learn
```

---

## ▶️ Usage

Open the notebook:

```bash
jupyter notebook vit_cats_vs_dogs.ipynb
```

Run all cells to:

1. Download & preprocess the dataset
2. Train the ViT model
3. Evaluate accuracy on test images

---

## 📊 Results

The Vision Transformer (ViT) model achieved excellent performance on the Cats vs Dogs dataset:

Training Loss (final epoch): 0.0000

* Validation Loss: 0.0160

* Validation Accuracy: 99.5%

* Validation F1-score: 0.995

* Test Accuracy: 99.49%

* Test F1-score: 0.995

These results show that ViT is highly effective at distinguishing between cats and dogs, achieving near-perfect classification performance on both validation and test sets

---


## 🔮 Future Improvements

* Try different pretrained backbones (e.g., Swin Transformer, ConvNeXt)
* Experiment with **larger input image sizes**
* Apply **hyperparameter tuning** for better results
* Deploy the model as a **web app (Streamlit/FastAPI)**

---

## 📝 License

This project is released under the MIT License.
