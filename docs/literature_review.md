# Literature Review & Verified Foundational References

## 1. Convolutional Siamese Change Detection
Early deep-learning remote sensing change detection adapted U-Net encoders into shared-weight Siamese backbones (`FC-Siam-conc` and `FC-Siam-diff`), demonstrating that comparing multi-scale feature differences outperforms raw pixel subtraction under illumination variations (Daudt et al., 2018).

## 2. Vision Transformers & Hierarchical Window Attention
Dosovitskiy et al. (2021) introduced the **Vision Transformer (ViT)**, modeling global spatial dependencies across flattened $16 \times 16$ patch embeddings via multi-head self-attention. Because quadratic token complexity and fixed single-scale patch grids challenge dense pixel segmentation, **Swin Transformer** (Liu et al., 2021) introduced hierarchical multi-resolution feature maps and linear-complexity shifted-window attention.

## 3. Transformer Change Detection & Benchmark Datasets
Chen & Shi (2020) introduced **LEVIR-CD** alongside spatial-temporal self-attention for high-resolution building change detection. Daudt et al. (2018) introduced **OSCD** for multispectral Sentinel-2 urban change monitoring. Subsequent remote sensing architectures such as **BIT-CD** (Chen et al., 2022) and **ChangeFormer** (Bandara & Patel, 2022) demonstrated strong long-range context modeling when paired with multi-scale decoders, though CNN baselines remain competitive under small-data regimes due to strong local inductive bias.

## 4. Explainability & Planetary-Scale Acquisition
Gradient-weighted Class Activation Mapping (**Grad-CAM**, Selvaraju et al., 2017) localizes spatial regions driving CNN activations, while **Attention Rollout** (Abnar & Zuidema, 2020) aggregates residual attention flow across Transformer layers. **Google Earth Engine** (Gorelick et al., 2017) provides planetary-scale access to analysis-ready Sentinel-2 collections (`COPERNICUS/S2_SR_HARMONIZED`).

---

## Verified References

1. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021).** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* International Conference on Learning Representations (ICLR 2021). arXiv: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
2. **Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021).** *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* IEEE/CVF International Conference on Computer Vision (ICCV 2021), pp. 9992–10002. DOI: [https://doi.org/10.1109/ICCV48922.2021.00986](https://doi.org/10.1109/ICCV48922.2021.00986)
3. **Daudt, R. C., Le Saux, B., & Boulch, A. (2018).** *Fully Convolutional Siamese Networks for Change Detection.* 25th IEEE International Conference on Image Processing (ICIP 2018). DOI: [https://doi.org/10.1109/ICIP.2018.8451652](https://doi.org/10.1109/ICIP.2018.8451652)
4. **Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018).** *Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks.* IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2018). arXiv DOI: [https://doi.org/10.48550/arXiv.1810.08468](https://doi.org/10.48550/arXiv.1810.08468)
5. **Chen, H., & Shi, Z. (2020).** *A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.* Remote Sensing, 12(10), 1662. DOI: [https://doi.org/10.3390/rs12101662](https://doi.org/10.3390/rs12101662)
6. **Chen, H., Qi, Z., & Shi, Z. (2022).** *Remote Sensing Image Change Detection With Transformers.* IEEE Transactions on Geoscience and Remote Sensing (TGRS). DOI: [https://doi.org/10.1109/TGRS.2021.3095166](https://doi.org/10.1109/TGRS.2021.3095166)
7. **Bandara, W. G. C., & Patel, V. M. (2022).** *A Transformer-Based Siamese Network for Change Detection.* IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2022). DOI: [https://doi.org/10.1109/IGARSS46834.2022.9883686](https://doi.org/10.1109/IGARSS46834.2022.9883686)
8. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* IEEE International Conference on Computer Vision (ICCV 2017). DOI: [https://doi.org/10.1109/ICCV.2017.74](https://doi.org/10.1109/ICCV.2017.74)
9. **Abnar, S., & Zuidema, W. (2020).** *Quantifying Attention Flow in Transformers.* Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020). DOI: [https://doi.org/10.18653/v1/2020.acl-main.385](https://doi.org/10.18653/v1/2020.acl-main.385)
10. **Gorelick, N., Hancher, M., Dixon, M., Ilyushchenko, S., Thau, D., & Moore, R. (2017).** *Google Earth Engine: Planetary-scale geospatial analysis for everyone.* Remote Sensing of Environment, 202, 18–27. DOI: [https://doi.org/10.1016/j.rse.2017.06.031](https://doi.org/10.1016/j.rse.2017.06.031)

