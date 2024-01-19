# iRNADis-PT

Identifying multiple types of disease-associated RNAs is of great significance for understanding the pathogenesis of disease. Recently, there is a surge of interest in identifying disease-associated RNAs via computational methods to overcome the challenge of tremendous manpower and material resources in biological experiments. However, current computational methods suffer from two main limitations: the limited identification capacity that only identify a single type of disease-associated RNAs and the lack of a systematic pathogenesis analysis that identified disease-associated RNAs are not applied to the downstream disease prognosis and therapeutic analysis. Here, we present **iRNADis-PT**, **a network-based framework that first utilizes heterogeneous graph learning to identify multiple types of disease-associated RNAs, including lncRNAs, snoRNAs, miRNAs, and mRNAs, and then apply these identified disease-associated RNAs to pathological analysis of disease prognosis and therapy**.

![iRNADis-PT](/imgs/iRNADis-PT.png)
**Fig. 1 Overall of the iRNADis-PT framework. a** workflow of data extraction and heterogeneous network construction. We first extract CNA data and expression information data from the TCGA database, based on which RNA features are extracted via copy number variation analysis and differential expression analysis. RNA similarities are calculated by the Pearson Correlation Coefficient to construct RNA networks. Next, we download RNA interactions and RNA-disease associations, and construct disease network. Finally, the heterogeneous network is constructed based on RNA networks, disease network, RNA interactions, and RNA-disease associations. **b** The overall framework of iRNADis-PT for identifying multiple types of disease-associated RNAs. The constructed heterogeneous network is fed into heterogeneous graph learning to extract node features, based on which we obtain RNA-disease association features inputting into MLP to predict candidate disease-associated RNAs. **c** Disease prognosis and therapy analysis. Based on predicted candidate disease-associated RNAs, we conduct differential expression analysis, classify risk groups, and drug sensitive analysis to explore the pathogenesis of disease.

# 1 Installation

## 1.1 Create conda environment

```
conda create -n iRNADis python=3.10
conda activate iRNADis
```

## 1.2 Requirements
The main dependencies used in this project are as follows (for more information, please see the `environment.yaml` file):

```
python  3.7.13
numpy  1.21.6
scipy  1.7.3
pandas  1.3.5
scikit-learn  1.0.2
torch  1.10.1+cu111
torch-geometric  2.1.0
tqdm  4.64.0
```

