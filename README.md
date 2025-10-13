# Sociodemographic Debiasing Influence on Fake News Detection

## Project Overview
This research project explores debiasing techniques for encoder models and studies their influence on fake news detection.  
The workflow is: (1) fine-tune an encoder (e.g., BERT) on a fake-news detection task to obtain a baseline; (2) implement and apply encoder debiasing techniques targeting sociodemographic attributes (e.g., race, gender, socioeconomic status); (3) evaluate and interpret the impact of debiasing on classification performance and fairness.

---

## Metadata

- **Institution:** Inria Paris  
- **Category:** Signal Processing / NLP  
- **Theme:** Debiasing algorithms' impact on misinformation detection  
- **Dataset:** LIAR dataset — Wang (2017). “Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection. (https://aclanthology.org/P17-2067/)  
- **Model (planned):** `bert-base-uncased` (Hugging Face)  
- **Type of organisation:** Research

---

## Supervisors / Contacts

- **Cecilia Graiff**  
  - Email: cecilia.graiff@inria.fr  
  - Phone: +39 348 158 5089

- **Gabrielle Le Bellier**  
  - Email: gabrielle.le-bellier@inria.fr  
  - Phone: +33 7 83 30 98 42

---

## Objectives
- Understand encoder-based models (BERT) and fine-tune them for fake-news detection.  
- Implement encoder debiasing techniques addressing protected / sociodemographic attributes.  
- Evaluate and compare model performance and fairness before and after debiasing.  
- Interpret trade-offs between accuracy and fairness for different debiasing approaches.

---

## Methodology (Suggested)
1. **Data preparation**
   - Download & preprocess the LIAR dataset.
   - Define labels and any auxiliary protected-attribute annotations required for debiasing experiments.

2. **Baseline**
   - Fine-tune `bert-base-uncased` for fake-news classification.
   - Evaluate with accuracy, precision, recall, F1, and confusion matrix.

3. **Debiasing techniques**
   - Implement and test several methods such as:
     - Iterative Nullspace Projection (INLP)
     - Adversarial debiasing (representation-level adversary)
     - Data-level balancing / augmentation
   - Apply debiasing to encoder representations (fine-tuned or frozen encoder as experimental variants).

4. **Evaluation & analysis**
   - Evaluate classification metrics and fairness metrics (e.g., demographic parity, equalized odds, subgroup performance).
   - Perform qualitative analysis on errors and model explanations when relevant.

---

## Tools & Libraries
- Hugging Face Transformers  
- PyTorch  
- scikit-learn  
- pandas  
- matplotlib (or other plotting libs)  

---

## Key References
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Wang, W. Y. (2017). *“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection.* ACL 2017. (LIAR dataset)
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020.

---

## Suggested Repo Structure

## Suggested Repo Structure

.
├── README.md
├── data/
│   ├── raw/                # raw LIAR dataset
│   └── processed/          # cleaned / tokenized data
├── notebooks/              # EDA and experiments
├── src/
│   ├── data.py             # data loading & preprocessing
│   ├── train.py            # training & evaluation scripts
│   ├── debiasing/          # implementations of debiasing methods
│   └── utils.py
├── experiments/            # logs, checkpoints, results
└── requirements.txt

---

## How to get started
1. Clone the repository.  
2. Install dependencies (e.g., `pip install -r requirements.txt`).  
3. Download the LIAR dataset and place it in `data/raw/`.  
4. Run preprocessing script: `python src/data.py`.  
5. Train baseline: `python src/train.py --config configs/baseline.yaml`.  
6. Run debiasing experiments: `python src/train.py --config configs/debias_inlp.yaml` (or other configs).

---

## License

MIT

---

## Acknowledgements
Project supervised at Inria Paris.
