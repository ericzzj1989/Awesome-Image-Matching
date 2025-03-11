# Awesome-Image-Matching

A curated list of resources for awesome imag matching related works. 

If you find some overlooked papers, please open issues or pull requests (recommended).

<p align="left">
    <a href="https://github.com/subeeshvasu/Awesome-Deblurring/pulls/new">Suggest new item</a>
    <br />
    <a href="https://github.com/subeeshvasu/Awesome-Deblurring/issues/new">Report Bug</a>
</p>

## Table of contents
- [Survey](#survey)
- [Benchmark](#benchmark)
- [Detector Learning](#detector-learning)
- [Descriptor Learning](#descriptor-learning)
- [Detector & Descriptor Learning](#detector--descriptor-learning)
- [Feature Matching](#feature-matching)
- [Others](#others)


## Survey
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2021|IJCV|[Image Matching from Handcrafted to Deep Features: A Survey](https://link.springer.com/content/pdf/10.1007/s11263-020-01359-2.pdf)|[Blog](https://blog.csdn.net/qq_42708183/article/details/109133806)|
|2023|arXiv|[Local Feature Matching Using Deep Learning: A Survey](https://arxiv.org/pdf/2401.17592.pdf)||

## Benchmark
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2005|TPAMI|[A Performance Evaluation of Local Descriptors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_pami2004.pdf)||
|2007|IJCV|[Automatic Panoramic Image Stitching using Invariant Features](http://matthewalunbrown.com/papers/ijcv2007.pdf)||
|2008|CVPR|[On Benchmarking Camera Calibration and Multi-View Stereo for High Resolution Imagery](https://homes.esat.kuleuven.be/~konijn/publications/2008/CS_cvpr_2008.pdf)||
|2011|ICCV|[Edge Foci Interest Points](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6126263)||
|2011|IJCV|[Interesting Interest Points](https://roboimagedata2.compute.dtu.dk/data/text/IJCV_2011.pdf)||
|2017|CVPR|[Comparative Evaluation of Hand-Crafted and Learned Local Features](https://demuc.de/papers/schoenberger2017comparative.pdf)|[Code](https://github.com/ahojnnes/local-feature-evaluation)|
|2017|CVPR|[HPatches: A benchmark and evaluation of handcrafted and learned local descriptors](https://openaccess.thecvf.com/content_cvpr_2017/papers/Balntas_HPatches_A_Benchmark_CVPR_2017_paper.pdf)|[Code & Project page](https://hpatches.github.io/)|
|2018|BMVC|[Large scale evaluation of local image feature detectors on homography datasets](http://www.bmva.org/bmvc/2018/contents/papers/0462.pdf)|[Code](https://github.com/lenck/vlb-deteval)|
|2018|CVPR|[MegaDepth: Learning Single-View Depth Prediction from Internet Photos](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_MegaDepth_Learning_Single-View_CVPR_2018_paper.pdf)|[Project page](https://www.cs.cornell.edu/projects/megadepth/)|
|2019|BMVC|[An Evaluation of Feature Matchers for Fundamental Matrix Estimation](https://jwbian.net/Papers/FM_BMVC19.pdf)|[Code](https://github.com/JiawangBian/FM-Bench)|
|2021|IJCV|[Image Matching across Wide Baselines: From Paper to Practice](https://arxiv.org/pdf/2003.01587.pdf)|[Code](https://github.com/ubc-vision/image-matching-benchmark)|
|2023|CVPR|[A Large Scale Homography Benchmark](https://arxiv.org/pdf/2302.09997.pdf)|[Code](https://github.com/danini/homography-benchmark)|
|2024|IROS|[GV-Bench: Benchmarking Local Feature Matching for Geometric Verification of Long-term Loop Closure Detection](https://arxiv.org/pdf/2407.11736)|[Code](https://github.com/jarvisyjw/GV-Bench)|
|2024|arXiv|[Mismatched: Evaluating the Limits of Image Matching Approaches and Benchmarks](https://arxiv.org/pdf/2408.16445)|[Code](https://github.com/surgical-vision/colmap-match-converter)|
|2024|arXiv|[Deep Learning Meets Satellite Images – An Evaluation on Handcrafted and Learning-based Features for Multi-date Satellite Stereo Images](https://arxiv.org/pdf/2409.02825)||
|2025|CVPR|[RUBIK: A Structured Benchmark for Image Matching across Geometric Challenges](https://arxiv.org/pdf/2502.19955)||


## Detector Learning
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2015|CVPR|[TILDE: A Temporally Invariant Learned DEtector](https://openaccess.thecvf.com/content_cvpr_2015/papers/Verdie_TILDE_A_Temporally_2015_CVPR_paper.pdf)|[Code](https://github.com/vcg-uvic/TILDE)|
|2016|ECCVW|[Learning Covariant Feature Detectors](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/lenc16learning.pdf)|[Code](https://github.com/lenck/ddet)|
|2017|CVPR|[Quad-networks: unsupervised learning to rank for interest point detection](https://www.microsoft.com/en-us/research/uploads/prod/2019/09/quad_cvpr17.pdf)||
|2017|CVPR|[Learning Discriminative and Transformation Covariant Local Feature Detectors](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Discriminative_and_CVPR_2017_paper.pdf)|[Code](https://github.com/ColumbiaDVMM/Transform_Covariant_Detector)|
|2018|CVPR|[Learning to Detect Features in Texture Images](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Learning_to_Detect_CVPR_2018_paper.pdf)|[Code](https://github.com/lg-zhang/pytorch-keypoint-release)|
|2019|ICCV|[Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters](http://refbase.cvc.uab.es/files/BRP2019.pdf)|[Code](https://github.com/axelBarroso/Key.Net)|
|2019|ICCV|[ELF: Embedded Localisation of Features in Pre-Trained CNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Benbihi_ELF_Embedded_Localisation_of_Features_in_Pre-Trained_CNN_ICCV_2019_paper.pdf)|[Code](https://github.com/abenbihi/elf)|
|2020|ACCV|[D2D: Keypoint Extraction with Describe to Detect Approach](https://openaccess.thecvf.com/content/ACCV2020/papers/Tian_D2D_Keypoint_Extraction_with_Describe_to_Detect_Approach_ACCV_2020_paper.pdf)||
|2021|WACV|[Learning of low-level feature keypoints for accurate and robust detection](https://openaccess.thecvf.com/content/WACV2021/papers/Suwanwimolkul_Learning_of_Low-Level_Feature_Keypoints_for_Accurate_and_Robust_Detection_WACV_2021_paper.pdf)||
|2022|CVPR|[Self-Supervised Equivariant Learning for Oriented Keypoint Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Self-Supervised_Equivariant_Learning_for_Oriented_Keypoint_Detection_CVPR_2022_paper.pdf)|[Project page](http://cvlab.postech.ac.kr/research/REKD/)|
|2023|ICCV|[NeSS-ST: Detecting Good and Stable Keypoints with a Neural Stability Score and the Shi-Tomasi detector](https://openaccess.thecvf.com/content/ICCV2023/papers/Pakulev_NeSS-ST_Detecting_Good_and_Stable_Keypoints_with_a_Neural_Stability_ICCV_2023_paper.pdf)|[Code](https://github.com/KonstantinPakulev/NeSS-ST)|
|2023|PRL|[Improving the matching of deformable objects by learning to detect keypoints](https://arxiv.org/pdf/2309.00434.pdf)|[Code](https://github.com/verlab/LearningToDetect_PRL_2023)|
|2024|WACV|[BALF: Simple and Efficient Blur Aware Local Feature Detector](https://openaccess.thecvf.com/content/WACV2024/papers/Zhao_BALF_Simple_and_Efficient_Blur_Aware_Local_Feature_Detector_WACV_2024_paper.pdf)|[Project page](https://ericzzj1989.github.io/balf)|
|2024|ICML|[Scale-Free Image Keypoints Using Differentiable Persistent Homology](https://arxiv.org/pdf/2406.01315)|[Code](https://github.com/gbarbarani/MorseDet)|
|2024|ECCV|[Learning to Make Keypoints Sub-Pixel Accurate](https://arxiv.org/pdf/2407.11668)|[Code](https://github.com/KimSinjeong/keypt2subpx)|
|2024|ECCV|[GMM-IKRS: Gaussian Mixture Models for Interpretable Keypoint Refinement and Scoring](https://arxiv.org/pdf/2408.17149)||
|2025|arXiv|[DaD: Distilled Reinforcement Learning for Diverse Keypoint Detection](https://arxiv.org/pdf/2503.07347)|[Code](https://github.com/parskatt/dad)|


## Descriptor Learning
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2012|ECCV|[Descriptor Learning Using Convex Optimisation](https://www.robots.ox.ac.uk/~vgg/publications/2012/Simonyan12/simonyan12.pdf)|[Code](https://github.com/cbalint13/opencv-dlco)|
|2012|NeurIPS|[Learning Image Descriptors with the Boosting-Trick](https://proceedings.neurips.cc/paper_files/paper/2012/file/0a09c8844ba8f0936c20bd791130d6b6-Paper.pdf)||
|2014|TPAMI|[Learning Local Feature Descriptors Using Convex Optimisation](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/simonyan14learning.pdf)||
|2014|arXiv|[Descriptor Matching with Convolutional Neural Networks: a Comparison to SIFT](https://lmb.informatik.uni-freiburg.de/Publications/2014/FDB14/1405.5769v1.pdf)||
|2015|CVPR|[Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/pdf/1504.03641.pdf)|[Blog](https://blog.csdn.net/m0_61899108/article/details/122609390)|
|2015|CVPR|[MatchNet: Unifying Feature and Metric Learning for Patch-Based Matching](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Han_MatchNet_Unifying_Feature_2015_CVPR_paper.pdf)|[Blog](https://blog.csdn.net/qq_36104364/article/details/115299866)|
|2015|ICCV|[Discriminative Learning of Deep Convolutional Feature Point Descriptors](https://openaccess.thecvf.com/content_iccv_2015/papers/Simo-Serra_Discriminative_Learning_of_ICCV_2015_paper.pdf)|[Code](https://github.com/etrulls/deepdesc-release)|
|2016|BMVC|[Learning local feature descriptors with triplets and shallow convolutional neural networks](http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf)|[Code](https://github.com/vbalnt/tfeat)|
|2016|CoRR|[PN-Net: Conjoined Triple Deep Networks for Learning Local Image Descriptors](https://arxiv.org/pdf/1601.05030.pdf)|[Code](https://github.com/vbalnt/pnnet), [Blog](https://blog.csdn.net/qq_36104364/article/details/115324732)|
|2016|CVPR|[Learning Local Image Descriptors with Deep Siamese and Triplet Convolutional Networks by Minimizing Global Loss Functions](https://openaccess.thecvf.com/content_cvpr_2016/papers/G_Learning_Local_Image_CVPR_2016_paper.pdf)||
|2016|CVPRW|[Euclidean and Hamming Embedding for Image Patch Description with Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w24/papers/Liu_Euclidean_and_Hamming_CVPR_2016_paper.pdf)||
|2016|RAL|[Self-Supervised Visual Descriptor Learning for Dense Correspondence](https://homes.cs.washington.edu/~tws10/3163.pdf)||
|2017|CVPR|[L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tian_L2-Net_Deep_Learning_CVPR_2017_paper.pdf)|[Code](https://github.com/yuruntian/L2-Net)|
|2017|NeurIPS|[Working hard to know your neighbor's margins: Local descriptor learning loss](https://arxiv.org/pdf/1705.10872.pdf)|[Code](https://github.com/DagnyT/hardnet)|
|2018|CoRL|[Leveraging Deep Visual Descriptors for Hierarchical Efficient Localization](https://arxiv.org/pdf/1809.01019.pdf)|[Code](https://github.com/ethz-asl/hierarchical_loc)|
|2018|CVPR|[Learning Deep Descriptors with Scale-Aware Triplet Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Keller_Learning_Deep_Descriptors_CVPR_2018_paper.pdf)|[Code](https://github.com/4uiiurz1/pytorch-scale-aware-triplet)|
|2018|CVPR|[Local Descriptors Optimized for Average Precision](https://openaccess.thecvf.com/content_cvpr_2018/papers/He_Local_Descriptors_Optimized_CVPR_2018_paper.pdf)||
|2018|ECCV|[GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zixin_Luo_Learning_Local_Descriptors_ECCV_2018_paper.pdf)|[Code](https://github.com/lzx551402/geodesc)|
|2019|CVPR|[ContextDesc: Local Descriptor Augmentation with Cross-Modality Context](https://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_ContextDesc_Local_Descriptor_Augmentation_With_Cross-Modality_Context_CVPR_2019_paper.pdf)|[Code](https://github.com/lzx551402/contextdesc)|
|2019|CVPR|[SOSNet: Second Order Similarity Regularization for Local Descriptor Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_SOSNet_Second_Order_Similarity_Regularization_for_Local_Descriptor_Learning_CVPR_2019_paper.pdf)|[Code](https://github.com/scape-research/SOSNet)|
|2019|CVWW|[Leveraging Outdoor Webcams for Local Descriptor Learning](https://arxiv.org/pdf/1901.09780.pdf)|[Data](https://github.com/pultarmi/AMOS_patches)|
|2019|ICCV|[Beyond Cartesian Representations for Local Descriptors](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ebel_Beyond_Cartesian_Representations_for_Local_Descriptors_ICCV_2019_paper.pdf)|[Code](https://github.com/cvlab-epfl/log-polar-descriptors)|
|2019|NeurIPS|[GIFT: Learning Transformation-Invariant Dense Visual Descriptors via Group CNNs](https://proceedings.neurips.cc/paper_files/paper/2019/file/34306d99c63613fad5b2a140398c0420-Paper.pdf)|[Project page](https://zju3dv.github.io/GIFT/)|
|2020|ECCV|[Learning Feature Descriptors using Camera Pose Supervision](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460732.pdf)|[Project page](https://qianqianwang68.github.io/CAPS/)|
|2020|ECCV|[Online Invariance Selection for Local Feature Descriptors](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470698.pdf)|[Code](https://github.com/rpautrat/LISRD)|
|2021|IROS|[RoRD: Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching](https://arxiv.org/pdf/2103.08573.pdf)|[Project page](https://uditsinghparihar.github.io/RoRD/)|
|2022|AAAI|[MTLDesc: Looking Wider to Describe Better](https://arxiv.org/pdf/2203.07003.pdf)|[Code](https://github.com/vignywang/MTLDesc)|
|2023|CVPR|[FeatureBooster: Boosting Feature Descriptors with a Lightweight Neural Network](https://arxiv.org/pdf/2211.15069.pdf)|[Code](https://github.com/SJTU-ViSYS/FeatureBooster)|
|2023|CVPR|[Learning Rotation-Equivariant Features for Visual Correspondence](https://arxiv.org/pdf/2303.15472.pdf)|[Project page](http://cvlab.postech.ac.kr/research/RELF/)|
|2023|TIM|[Illumination-insensitive Binary Descriptor for Visual Measurement Based on Local Inter-patch Invariance](https://arxiv.org/pdf/2305.07943.pdf)||
|2023|ICRA|[Descriptor Distillation for Efficient Multi-Robot SLAM](https://arxiv.org/pdf/2303.08420.pdf)||
|2023|arXiv|[Residual Learning for Image Point Descriptors](https://arxiv.org/pdf/2312.15471.pdf)||
|2024|CVPR|[Steerers: A framework for rotation equivariant keypoint descriptors](https://arxiv.org/pdf/2312.02152)|[Code](https://github.com/georg-bn/rotation-steerers)|
|2024|ECCV|[Affine steerers for structured keypoint description](https://arxiv.org/pdf/2408.14186)|[Code](https://github.com/georg-bn/affine-steerers)|
|2025|arXiv|[PromptMID: Modal Invariant Descriptors Based on Diffusion and Vision Foundation Models for Optical-SAR Image Matching](https://arxiv.org/pdf/2502.18104)|[Code](https://github.com/HanNieWHU/PromptMID)|


## Detector & Descriptor Learning
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2016|ECCV|[LIFT: Learned Invariant Feature Points](https://arxiv.org/pdf/1603.09114.pdf)|[Code](https://github.com/cvlab-epfl/LIFT)|
|2018|CVPRW|[SuperPoint: Self-Supervised Interest Point Detection and Description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf)|[Code](https://github.com/magicleap/SuperPointPretrainedNetwork)|
|2018|NeurIPS|[LF-Net: Learning Local Features from Images](https://proceedings.neurips.cc/paper_files/paper/2018/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf)|[Code](https://github.com/vcg-uvic/lf-net-release)|
|2019|CVPR|[D2-Net: A Trainable CNN for Joint Detection and Description of Local Features](https://openaccess.thecvf.com/content_CVPR_2019/papers/Dusmanu_D2-Net_A_Trainable_CNN_for_Joint_Description_and_Detection_of_CVPR_2019_paper.pdf)|[Project page](https://github.com/mihaidusmanu/d2-net)|
|2019|NeurIPS|[R2D2: Repeatable and Reliable Detector and Descriptor](https://proceedings.neurips.cc/paper_files/paper/2019/file/3198dfd0aef271d22f7bcddd6f12f5cb-Paper.pdf)|[Code](https://github.com/naver/r2d2)|
|2019|arXiv|[UnsuperPoint: End-to-end Unsupervised Interest Point Detector and Descriptor](https://arxiv.org/pdf/1907.04011.pdf)||
|2020|CVPR|[ASLFeat: Learning Local Features of Accurate Shape and Localization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Luo_ASLFeat_Learning_Local_Features_of_Accurate_Shape_and_Localization_CVPR_2020_paper.pdf)|[Code](https://github.com/lzx551402/aslfeat)|
|2020|CVPR|[Reinforced Feature Points: Optimizing Feature Detection and Description for a High-Level Task](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bhowmik_Reinforced_Feature_Points_Optimizing_Feature_Detection_and_Description_for_a_CVPR_2020_paper.pdf)|[Code](https://github.com/aritra0593/Reinforced-Feature-Points)|
|2020|ICLR|[Neural Outlier Rejection for Self-Supervised Keypoint Learning](https://openreview.net/pdf?id=Skx82ySYPH)|[Code](https://github.com/TRI-ML/KP2D)|
|2020|NeurIPS|[DISK: Learning local features with policy gradient](https://proceedings.neurips.cc//paper/2020/file/a42a596fc71e17828440030074d15e74-Paper.pdf)|[Code](https://github.com/cvlab-epfl/disk)|
|2020|arXiv|[SEKD: Self-Evolving Keypoint Detection and Description](https://arxiv.org/pdf/2006.05077.pdf)||
|2021|IROS|[RaP-Net: A Region-wise and Point-wise Weighting Network to Extract Robust Features for Indoor Localization](https://arxiv.org/pdf/2012.00234.pdf)|[Code](https://github.com/ivipsourcecode/RaP-Net)|
|2022|ACCV|[Rethinking Low-level Features for Interest Point Detection and Description](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_Rethinking_Low-level_Features_for_Interest_Point_Detection_and_Description_ACCV_2022_paper.pdf)|[Code](https://github.com/wangch-g/lanet)|
|2022|BMVC|[Local Feature Extraction from Salient Regions by Feature Map Transformation]()|[Project page](https://bmvc2022.mpi-inf.mpg.de/552/)|
|2022|CVPR|[Decoupling Makes Weakly Supervised Local Feature Better](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Decoupling_Makes_Weakly_Supervised_Local_Feature_Better_CVPR_2022_paper.pdf)|[Code](https://github.com/The-Learning-And-Vision-Atelier-LAVA/PoSFeat)|
|2022|ECCV|[Semi-Supervised Keypoint Detector and Descriptor for Retinal Image Matching](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136810586.pdf)|[Code](https://github.com/ruc-aimc-lab/SuperRetina)|
|2022|NeurIPS|[TUSK: Task-Agnostic Unsupervised Keypoints](https://arxiv.org/pdf/2206.08460.pdf)||
|2022|TMM|[ALIKE: Accurate and Lightweight Keypoint Detection and Descriptor Extraction](https://arxiv.org/pdf/2112.02906.pdf)|[Code](https://github.com/Shiaoming/ALIKE)|
|arXiv|2022|[Shared Coupling-bridge for Weakly Supervised Local Feature Learning](https://arxiv.org/pdf/2212.07047.pdf)|[Code](https://github.com/sunjiayuanro/SCFeat)|
|2023|AAAI|[DarkFeat: Noise-Robust Feature Detector and Descriptor for Extremely Low-Light RAW Images](https://ojs.aaai.org/index.php/AAAI/article/view/25161/24933)|[Code](https://github.com/THU-LYJ-Lab/DarkFeat)|
|2023|CVPR|[Enhancing Deformable Local Features by Jointly Learning to Detect and Describe Keypoints](https://arxiv.org/pdf/2304.00583.pdf)|[Project page](https://www.verlab.dcc.ufmg.br/descriptors/dalf_cvpr23/)|
|2023|CVPR|[SFD2: Semantic-Guided Feature Detection and Description](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_SFD2_Semantic-Guided_Feature_Detection_and_Description_CVPR_2023_paper.pdf)|[Code](https://github.com/feixue94/sfd2)|
|2023|CVPR|[Learning Transformation-Predictive Representations for Detection and Description of Local Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Learning_Transformation-Predictive_Representations_for_Detection_and_Description_of_Local_Features_CVPR_2023_paper.pdf)||
|2023|CVPR|[D2Former: Jointly Learning Hierarchical Detectors and Contextual Descriptors via Agent-Based Transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/He_D2Former_Jointly_Learning_Hierarchical_Detectors_and_Contextual_Descriptors_via_Agent-Based_CVPR_2023_paper.pdf)||
|2023|CVPR|[Learning Transformation-Predictive Representations for Detection and Description of Local Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Learning_Transformation-Predictive_Representations_for_Detection_and_Description_of_Local_Features_CVPR_2023_paper.pdf)||
|2023|CVPRW|[ZippyPoint: Fast Interest Point Detection, Description, and Matching through Mixed Precision Discretization](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Kanakis_ZippyPoint_Fast_Interest_Point_Detection_Description_and_Matching_Through_Mixed_CVPRW_2023_paper.pdf)|[Code](https://github.com/menelaoskanakis/ZippyPoint)|
|2023|RAL|[Learning Task-Aligned Local Features for Visual Localization](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10103591)||
|2023|TIM|[ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation](https://arxiv.org/pdf/2304.03608.pdf)|[Code](https://github.com/Shiaoming/ALIKED)|
|2023|TPAMI|[Attention Weighted Local Descriptors](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10105519)|[Code](https://github.com/vignywang/AWDesc)|
|2023|ICCV|[SiLK: Simple Learned Keypoints](https://arxiv.org/pdf/2304.06194.pdf)|[Code](https://github.com/facebookresearch/silk)|
|2023|ICCV|[S-TREK: Sequential Translation and Rotation Equivariant Keypoints for local feature extraction](https://arxiv.org/pdf/2308.14598.pdf)||
|2023|ICRA|[Knowledge Distillation for Feature Extraction in Underwater VSLAM](https://arxiv.org/pdf/2303.17981.pdf)|[Dataset](https://github.com/Jinghe-mel/UFEN-SLAM)|
|2024|3DV|[DeDoDe: Detect, Don't Describe -- Describe, Don't Detect for Local Feature Matching](https://arxiv.org/pdf/2308.08479.pdf)|[Code](https://github.com/Parskatt/DeDoDe)|
|2024|CVPR|[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/pdf/2404.19174)|[Project page](https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/)|
|2024|CVPRW|[DeDoDe v2: Analyzing and Improving the DeDoDe Keypoint Detector](https://openaccess.thecvf.com/content/CVPR2024W/IMW/papers/Edstedt_DeDoDe_v2_Analyzing_and_Improving_the_DeDoDe_Keypoint_Detector_CVPRW_2024_paper.pdf)|[Code](https://github.com/Parskatt/DeDoDe)|
|2024|arXiv|[LBurst: Learning-Based Robotic Burst Feature Extraction for 3D Reconstruction in Low Light](https://arxiv.org/pdf/2410.23522)|[Project page](https://roboticimaging.org/Projects/LBurst/)|
|2025|WACV|[EI-Nexus: Towards Unmediated and Flexible Inter-Modality Local Feature Extraction and Matching for Event-Image Data](https://arxiv.org/pdf/2410.21743)|[Code](https://github.com/ZhonghuaYi/EI-Nexus_official)|


## Feature Matching
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2020|CVPR|[SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/pdf/1911.11763)|[Code](https://github.com/magicleap/SuperGluePretrainedNetwork)|
|2021|CVPR|[LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/pdf/2104.00680)|[Project page](https://zju3dv.github.io/loftr/)|
|2021|ICCV|[COTR: Correspondence Transformer for Matching Across Images](https://arxiv.org/pdf/2103.14167)|[Code](https://github.com/ubc-vision/COTR)|
|2022|ICLR|[QuadTree Attention for Vision Transformers](https://arxiv.org/pdf/2201.02767)|[Code](https://github.com/Tangshitao/QuadTreeAttention)|
|2022|ECCV|[ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer](https://arxiv.org/pdf/2208.14201)|[Code](https://github.com/apple/ml-aspanformer)|
|2022|ACCV|[MatchFormer: Interleaving Attention in Transformers for Feature Matching](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_MatchFormer_Interleaving_Attention_in_Transformers_for_Feature_Matching_ACCV_2022_paper.pdf)|[Code](https://github.com/jamycheung/MatchFormer)|
|2023|CVPR|[DKM: Dense Kernelized Feature Matching for Geometry Estimation](https://arxiv.org/pdf/2202.00667)|[Project page](https://parskatt.github.io/DKM/)|
|2023|CVPR|[IMP: Iterative Matching and Pose Estimation with Adaptive Pooling](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_IMP_Iterative_Matching_and_Pose_Estimation_With_Adaptive_Pooling_CVPR_2023_paper.pdf)|[Code](https://github.com/feixue94/imp-release)|
|2023|ICCV|[LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/pdf/2306.13643)|[Code](https://github.com/cvg/LightGlue)|
|2023|ICCV|[End2End Multi-View Feature Matching with Differentiable Pose Optimization](https://arxiv.org/pdf/2205.01694)|[Project page](https://barbararoessle.github.io/e2e_multi_view_matching/)|
|2023|arXiv|[Searching from Area to Point: A Semantic Guided Framework with Geometric Consistency for Accurate Feature Matching](https://arxiv.org/pdf/2305.00194)|[Code](https://github.com/Easonyesheng/SGAM)|
|2024|CVPR|[RoMa: Robust Dense Feature Matching](https://arxiv.org/pdf/2305.15404)|[Project page](https://parskatt.github.io/RoMa/)|
|2024|CVPR|[OmniGlue: Generalizable Feature Matching with Foundation Model Guidance](https://arxiv.org/pdf/2405.12979)|[Project page](https://hwjiang1510.github.io/OmniGlue/)|
|2024|CVPR|[Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse-Like Speed](https://zju3dv.github.io/efficientloftr/files/EfficientLoFTR.pdf)|[Project page](https://zju3dv.github.io/efficientloftr/)|
|2024|CVPR|[MESA: Matching Everything by Segmenting Anything](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_MESA_Matching_Everything_by_Segmenting_Anything_CVPR_2024_paper.pdf)|[Code](https://github.com/Easonyesheng/A2PM-MESA)|
|2024|CVPRW|[XoFTR: Cross-modal Feature Matching Transformer](https://openaccess.thecvf.com/content/CVPR2024W/IMW/papers/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.pdf)|[Code](https://github.com/OnderT/XoFTR)|
|2024|ECCV|[Raising the Ceiling: Conflict-Free Local Feature Matching with Dynamic View Switching](https://arxiv.org/pdf/2407.07789)||
|2024|ECCV|[StereoGlue: Robust Estimation with Single-Point Solvers](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07485.pdf)|[Code](https://github.com/danini/stereoglue)|
|2024|ECCV|[iMatching: Imperative Correspondence Learning](https://arxiv.org/pdf/2312.02141)|[Code](https://github.com/sair-lab/iMatching)|
|2024|MM|[PRISM: PRogressive dependency maxImization for Scale-invariant image Matching](https://arxiv.org/pdf/2408.03598)|[Code](https://github.com/Master-cai/PRISM)|
|2024|ACCV|[Leveraging Semantic Cues from Foundation Vision Models for Enhanced Local Feature Correspondence](https://arxiv.org/pdf/2410.09533)|[Project page](https://www.verlab.dcc.ufmg.br/descriptors/reasoning_accv24/)|
|2024|NeurIPS|[ETO: Efficient Transformer-based Local Feature Matching by Organizing Multiple Homography Hypotheses](https://arxiv.org/pdf/2410.22733)||
|2024|IJCV|[Feature Matching via Graph Clustering with Local Affine Consensus](https://link.springer.com/article/10.1007/s11263-024-02291-5)||
|2023|arXiv|[AffineGlue: Joint Matching and Robust Estimation](https://arxiv.org/pdf/2307.15381)||
|2024|arXiv|[Grounding Image Matching in 3D with MASt3R](https://arxiv.org/pdf/2406.09756)|[Code](https://github.com/naver/mast3r)|
|2024|arXiv|[DMESA: Densely Matching Everything by Segmenting Anything](https://arxiv.org/pdf/2408.00279)|[Code](https://github.com/Easonyesheng/A2PM-MESA)|
|2024|arXiv|[ConDL: Detector-Free Dense Image Matching](https://arxiv.org/pdf/2408.02766)||
|2024|arXiv|[Geometry-aware Feature Matching for Large-Scale Structure from Motion](https://arxiv.org/pdf/2409.02310)||
|2024|arXiv|[HomoMatcher: Dense Feature Matching Results with Semi-Dense Efficiency by Homography Estimation](https://arxiv.org/pdf/2411.06700)||
|2024|arXiv|[GIMS: Image Matching System Based on Adaptive Graph Construction and Graph Neural Network](https://arxiv.org/pdf/2412.18221)|[Code](https://github.com/songxf1024/GIMS)|
|2024|arXiv|[MINIMA: Modality Invariant Image Matching](https://arxiv.org/pdf/2412.19412)|[Code](https://github.com/LSXI7/MINIMA)|
|2024|arXiv|[XPoint: A Self-Supervised Visual-State-Space based Architecture for Multispectral Image Registration](https://arxiv.org/pdf/2411.07430)|[Code](https://github.com/canyagmur/XPoint)|
|2025|ICRA|[MambaGlue: Fast and Robust Local Feature Matching With Mamba](https://arxiv.org/pdf/2502.00462)|[Code](https://github.com/url-kaist/MambaGlue)|
|2025|CVPR|[EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching](https://arxiv.org/pdf/2502.20685)|[Project page](https://jdk9405.github.io/EDM/)|
|2025|CVPR|[JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](https://arxiv.org/pdf/2503.03437)|[Project page](https://leoluxxx.github.io/JamMa-page/)|
|2025|arXiv|[MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training](https://arxiv.org/pdf/2501.07556)|[Project page](https://zju3dv.github.io/MatchAnything/)|
|2025|arXiv|[MIFNet: Learning Modality-Invariant Features for Generalizable Multimodal Image Matching](https://arxiv.org/pdf/2501.11299)||
|2025|arXiv|[Diff-Reg v2: Diffusion-Based Matching Matrix Estimation for Image Matching and 3D Registration](https://arxiv.org/pdf/2503.04127)||
|2025|arXiv|[EDM: Efficient Deep Feature Matching](https://arxiv.org/pdf/2503.05122)|[Code](https://github.com/chicleee/EDM)|


## Others
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2016|CVPR|[Learning to Assign Orientations to Feature Points](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yi_Learning_to_Assign_CVPR_2016_paper.pdf)|[Code](https://github.com/vcg-uvic/benchmark-orientation)|
|2020|CVPR|[On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kayhan_On_Translation_Invariance_in_CNNs_Convolutional_Layers_Can_Exploit_Absolute_CVPR_2020_paper.pdf)|[Code](https://github.com/oskyhn/CNNs-Without-Borders)|
|2022|AAAI|[Guide Local Feature Matching by Overlap Estimation](https://arxiv.org/pdf/2202.09050)|[Code](https://github.com/AbyssGaze/OETR)|
|2024|ICLR|[GIM: Learning Generalizable Image Matcher From Internet Videos](https://arxiv.org/pdf/2402.11095)|[Project page](https://xuelunshen.com/gim/)|
