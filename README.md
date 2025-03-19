# Truck Emissions Analysis with Interpretable Machine Learning

This work explores the impact of the built environment on truck emissions using interpretable machine learning, accounting for spatiotemporal and policy heterogeneity.

## Overview

This research develops a comprehensive framework to analyze the nonlinear effects of the built environment on heavy-duty diesel truck emissions. The study utilizes large-scale GPS data from Shanghai, China, integrating random effects with a light gradient boosting machine (LightGBM) to account for spatiotemporal and policy influences.

## Key Features

- Interpretable machine learning model combining random effects with LightGBM
- Analysis of nonlinear relationships between built environment factors and truck emissions
- Comprehensive performance evaluation across different spatial and temporal dimensions
- Visualization of model performance by hour, day period, district, and other groupings

## Methodology

The study employs a Mixed Effects approach combined with various machine learning models (SVR, XGBoost, LightGBM) to capture both fixed effects from built environment variables and random effects from spatiotemporal and policy factors. The model accounts for:

- Spatial heterogeneity (districts, zones)
- Temporal patterns (hour of day, day period)
- Policy restrictions (PRU - Policy Restriction Units)

## Citation

If you use this code or methodology in your research, please cite the original paper:

```bibtex
@article{SHI2025104662,
title = {Revealing the built environment impacts on truck emissions using interpretable machine learning},
journal = {Transportation Research Part D: Transport and Environment},
volume = {141},
pages = {104662},
year = {2025},
issn = {1361-9209},
doi = {https://doi.org/10.1016/j.trd.2025.104662},
url = {https://www.sciencedirect.com/science/article/pii/S1361920925000720},
author = {Tongtong Shi and Meiting Tu and Ye Li and Haobing Liu and Dominique Gruyer},
keywords = {Urban freight transport, Truck emissions, Built environment, Nonlinear effects, Interpretable machine learning}
