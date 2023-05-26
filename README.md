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
- [Others](#others)


## Survey
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2021|IJCV|[Image Matching from Handcrafted to Deep Features: A Survey](https://link.springer.com/content/pdf/10.1007/s11263-020-01359-2.pdf)|[Blog](https://blog.csdn.net/qq_42708183/article/details/109133806)|

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


## Descriptor Learning
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2012|ECCV|[Descriptor Learning Using Convex Optimisation](https://www.robots.ox.ac.uk/~vgg/publications/2012/Simonyan12/simonyan12.pdf)|[Code](https://github.com/cbalint13/opencv-dlco)|
|2012|NeurIPS|[Learning Image Descriptors with the Boosting-Trick](https://proceedings.neurips.cc/paper_files/paper/2012/file/0a09c8844ba8f0936c20bd791130d6b6-Paper.pdf)||
|2014|TPAMI|[Learning Local Feature Descriptors Using Convex Optimisation](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/simonyan14learning.pdf)||
|2014|ArXiv|[Descriptor Matching with Convolutional Neural Networks: a Comparison to SIFT](https://lmb.informatik.uni-freiburg.de/Publications/2014/FDB14/1405.5769v1.pdf)||
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
|2023|arXiv|[Descriptor Distillation for Efficient Multi-Robot SLAM](https://arxiv.org/pdf/2303.08420.pdf)||


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
|2023|CVPR|[Enhancing Deformable Local Features by Jointly Learning to Detect and Describe Keypoints](https://arxiv.org/pdf/2304.00583.pdf)|[Project page](https://www.verlab.dcc.ufmg.br/descriptors/dalf_cvpr23/)|
|2023|CVPR|[SFD2: Semantic-Guided Feature Detection and Description](https://openaccess.thecvf.com/content/CVPR2023/papers/Xue_SFD2_Semantic-Guided_Feature_Detection_and_Description_CVPR_2023_paper.pdf)|[Code](https://github.com/feixue94/sfd2)|
|2023|CVPR|[Learning Transformation-Predictive Representations for Detection and Description of Local Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Learning_Transformation-Predictive_Representations_for_Detection_and_Description_of_Local_Features_CVPR_2023_paper.pdf)||
|2023|CVPR|[D2Former: Jointly Learning Hierarchical Detectors and Contextual Descriptors via Agent-Based Transformers](https://openaccess.thecvf.com/content/CVPR2023/papers/He_D2Former_Jointly_Learning_Hierarchical_Detectors_and_Contextual_Descriptors_via_Agent-Based_CVPR_2023_paper.pdf)||
|2023|CVPR|[Learning Transformation-Predictive Representations for Detection and Description of Local Features](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Learning_Transformation-Predictive_Representations_for_Detection_and_Description_of_Local_Features_CVPR_2023_paper.pdf)||
|2023|RAL|[Learning Task-Aligned Local Features for Visual Localization](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10103591)||
|2023|TIM|[ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation](https://arxiv.org/pdf/2304.03608.pdf)|[Code](https://github.com/Shiaoming/ALIKED)|
|2023|TPAMI|[Attention Weighted Local Descriptors](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10105519)|[Code](https://github.com/vignywang/AWDesc)|
|2023|arXiv|[SiLK: Simple Learned Keypoints](https://arxiv.org/pdf/2304.06194.pdf)|[Code](https://github.com/facebookresearch/silk)|


## Others
|Year|Pub|Paper|Repo|
|:---:|:---:|:---:|:---:|
|2016|CVPR|[Learning to Assign Orientations to Feature Points](https://openaccess.thecvf.com/content_cvpr_2016/papers/Yi_Learning_to_Assign_CVPR_2016_paper.pdf)||
|2020|CVPR|[On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kayhan_On_Translation_Invariance_in_CNNs_Convolutional_Layers_Can_Exploit_Absolute_CVPR_2020_paper.pdf)|[Code](https://github.com/oskyhn/CNNs-Without-Borders)|
