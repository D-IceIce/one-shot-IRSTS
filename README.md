# One Shot is Enough for Sequential Infrared Small Target Segmentation

## Introduction

The code in this repository is the implementation of the methods 
and algorithms described in the paper "[One Shot is Enough for Sequential Infrared Small Target Segmentation](https://arxiv.org/abs/2408.04823)".
For detailed theoretical background and methodology, please refer to the paper.

## Getting Started

To use this code, follow these steps:

1. Download the [IRDST](http://xzbai.buaa.edu.cn/datasets.html) dataset and place it in the `dataset` folder.

2. Download the weight of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and place it in the `weights` folder.

3. Install required Libraries.

4. Run the `main.py` file.

## Acknowledgements
We are grateful for the resources that have supported this work:

- SAM-related code from the [segment-anything](https://github.com/facebookresearch/segment-anything/tree/main) project.
- MobileSAM weights available at [MobileSAM](https://github.com/ChaoningZhang/MobileSAM).

Thanks to the contributors of these projects for their invaluable contributions.

## Contact

For more information or inquiries regarding the one-shot-IRSTS model and its applications, please contact [danbingbing20@mails.ucas.ac.cn].
