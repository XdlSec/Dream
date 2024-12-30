# DREAM for Combating Concept Drift

Code for paper `Combating Concept Drift with Explanatory Detection and Adaptation for Android Malware Classification`.
This paper is currently under submission. We have concealed two files in the main code (`/dream`) and will release the complete code upon acceptance.

## Setup

``` python
conda create --name dream python=3.8
conda activate dream
conda install conda-forge::tensorflow-gpu=2.11.0 # for CUDA 12.5
conda install -c conda-forge libgcc-ng libstdcxx-ng
conda install pandas matplotlib scikit-learn tqdm --yes
```
