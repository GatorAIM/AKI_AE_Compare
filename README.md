# Autoencoder-Based Representation Learning for Similar Patient Retrieval from Electronic Health Records: A Comparative Study

## Background
Analyzing electronic health record (EHR) snapshots of similar patients enables physicians to proactively predict disease onset, customize treatment plans, and anticipate patient-specific trajectories. However, EHR data are inherently challenging to model due to their high dimensionality, mixed feature types, noise, bias, and sparsity. 

Autoencoder (AE)-based patient representation learning presents promising opportunities to address these challenges. However, a critical question remains: how do different AE designs and distance measures impact the quality of retrieved similar patient cohorts?

## Objective
This study aims to evaluate the performance of five common AE variants in retrieving similar patients:

- Vanilla Autoencoder (AE)
- Denoising Autoencoder (DAE)
- Contractive Autoencoder (CAE)
- Sparse Autoencoder (SAE)
- Robust Autoencoder (RAE)

Additionally, the study investigates the impact of different distance measures and hyperparameter configurations on model performance.

## Method
We tested these five AE variants on two real-world datasets:

- University of Kansas Medical Center (KUMC, n = 13,752)
- Medical College of Wisconsin (MCW, n = 9,568)

In total, we evaluated 168 different hyperparameter configurations. 

To retrieve similar patients, we applied **Euclidean distance-based k-nearest neighbors (k-NN)** and **Mahalanobis distance-based k-NN** on the latent representations learned by the autoencoders.

We evaluated model performance on two clinical prediction tasks:

1. **Acute Kidney Injury (AKI) onset prediction**
2. **Post-discharge 1-year mortality prediction**

**F1 score** was used as the primary evaluation metric.

## Results
Key findings from our study include:

1. **DAE outperformed other AE variants when paired with Euclidean distance (P<.001), followed by Vanilla AE and CAE.**
2. **Learning rates significantly influenced the performance of AE variants.**
3. **Mahalanobis distance-based k-NN often outperformed Euclidean distance-based k-NN when applied to latent representations.**

However, we also found that whether AE models improved performance by transforming raw data into latent representations—compared to directly applying Mahalanobis distance-based k-NN on raw data—was data-dependent.

## Conclusion
This study provides a comprehensive analysis of the performance of different AE variants for similar patient retrieval. It also evaluates the impact of various hyperparameter configurations on model performance.

The findings lay a solid foundation for future research into AE-based patient similarity estimation and the development of personalized medicine models.
