# MIP supported RiverML

![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## **This is a repository for deployment of Bankfull TopWidth(TW), Depth(Y), and shape (r)**

A machine learning approach for estimation of bankfull width and depth 

- [Repository](#repository)
  - [Cloning](#cloning)
  - [Data and Model](#data-and-model)
  - [Deployment](#deployment)

## Repository

### Cloning

```shell
git clone https://github.com/lynker-spatial/RiverShapeML.git
```

### Data and Model

Data are located in data folder, and models are locatred in models all are compressed and split to chunks. The bash script handels reassmebling and decompression.

### Deployment

To deploy the ML model 

```shell
chmod u+x deploy.bash
bash deploy.bash -n -1 
```
Where:  

**-n** is the number of cores to be used in parallel. An integer depends on the number of cores. Use -1 for utilizing all

This command will: <br>
1- Check for system dependencies and reassemble the necessary data and model files. <br>
2- Install Miniconda if it's not already present. <br>
3- Create and activate a dedicated Conda environment (WD-env) from the wd_env.yaml file. <br>
4- Run the inference.py script to perform the estimations. <br>

Output:
All output from the inference process, including any status messages and errors, will be saved to a file named inference_output.out.
