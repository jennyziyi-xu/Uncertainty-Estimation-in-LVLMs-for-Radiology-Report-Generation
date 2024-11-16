# Uncertainty Estimation in Large Vision Language Models for Automated Radiology Report Generation

## Abstract

The automated generation of free-text radiology reports is crucial for improving diagnosis and treatment in clinical practice. The latest chest X-ray report generation models utilize large vision language model (LVLM) architectures, which demand a higher level of interpretability for clinical deployment. Uncertainty estimation scores can assist clinicians in evaluating the reliability of these model outputs and promoting broader adoption of automated systems. In this paper, we conduct a comprehensive evaluation of the correlation between 16 LLM uncertainty scores and 6 radiology report evaluation metrics across 4 state-of-the-art LVLMs for CXR report generation. Our findings show a strong Pearson correlation, ranging from 0.4 to 0.6 on a scale from -1 to 1, for several models. We provide a detailed analysis of these uncertainty scores and evaluation metrics, offering insights in applying these methods in real clinical settings. This study is the first to evaluate LLM-based uncertainty estimation scores for X-ray report generation LVLM models, establishing a benchmark and laying the groundwork for their adoption in clinical practice.

## Installation

Basic requirements are:
- transformers
- numpy
- pandas
- matplotlib
- seaborn
- torch

To compute perturbation-based and sample-based uncertainty scores, install [sentence-transformers](https://github.com/UKPLab/sentence-transformers?tab=readme-ov-file)

## Pipeline to benchmark uncertainty estimation scores against evaluation scores. 

1, Use the functionalities in ```model_utils/base.py``` and adapt to specific VLMs to generate output logits in a csv. format.

2, Run model inference and use [CXR-Report-Metric Github repo](https://github.com/rajpurkarlab/CXR-Report-Metric) to compute the evaluation metrics such as RadGraph. 

3, ```scripts/uncertainty_score.py``` contains the functionalities to compute single-inference uncertainty scores. 

4, Run stochastic model inferences and use ```scripts/sampled-based-score.py``` to compute sample-based uncertainty scores. 

5, Use ```scripts/perturbation.py``` to find the points of perturbation. 

6, Run model inferences based on the computed points of perturbation. Use ```scripts/perturbation_based_score.py``` to compute perturbation-based uncertainty scores. 

7, Obtain Pearson Correlation coefficients for each pair of evaluation metric and uncertainty score, and plot them using ```correlations/plot_result.py```. 

8, Analyze the effect of filtering out evaluation metric or uncertainty score from the median, and compute Pearson, Kendall's Tau and Spearman correlations: ```correlations/all_correlation.py```. 