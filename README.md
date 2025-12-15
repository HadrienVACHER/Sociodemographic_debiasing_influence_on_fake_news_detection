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

- **Gabrielle Le Bellier**  
  - Email: gabrielle.le-bellier@inria.fr

---

## Description

This project aims to explore debiasing techniques for encoder models and study their influence on fake news detection. The initial phase involves analyzing the mechanics of encoder models, such as BERT, and fine-tuning a classifier to establish a baseline performance on a fake news detection task. Subsequently, well-known encoder debiasing techniques regarding discriminatory attributes (e.g., race, gender, socioeconomic status) will be implemented and studied. Finally, the debiased models will be evaluated against the baseline, with the results interpreted based on the specific mitigation techniques applied.

---

## Objectives
- Understand encoder-based models (BERT) and fine-tune them for fake-news detection.  
- Implement encoder debiasing techniques addressing protected / sociodemographic attributes.  
- Evaluate and compare model performance and fairness before and after debiasing.  
- Interpret trade-offs between accuracy and fairness for different debiasing approaches.

---

## Methodology

x

---

## Tools & Libraries
- Hugging Face Transformers  
- PyTorch  
- scikit-learn  
- pandas  
- matplotlib 

---

## Key References
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019.
- Wang, W. Y. (2017). *“Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News Detection.* ACL 2017. (LIAR dataset)
- Ravfogel, S., Elazar, Y., Gonen, H., Twiton, M., & Goldberg, Y. (2020). *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020.

---

## Repo Structure

```
├── README.md
├── Bibliography
├── Code
├── Data
```
---

## License

This project is licensed under the MIT License. See [LICENSE](Stat_App/LICENSE) for details.

---

## Acknowledgements
Project supervised at Inria Paris.
