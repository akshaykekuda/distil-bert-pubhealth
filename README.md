# Explainable Automated Fact-Checking for Public Health Claims

This repository aims to solve the problem of fact checking of public health claims using distilbert transformer model. 

## About the Data

PUBHEALTH is a comprehensive dataset for explainable automated fact-checking of public health claims. Each instance in the PUBHEALTH dataset has an associated veracity label (true, false, unproven, mixture). Furthermore each instance in the dataset has an explanation text field. The explanation is a justification for which the claim has been assigned a particular veracity label.

## Data Preparation: 
On close observation of the dataset, I found some values of labels to be -1. Since the dataset has only 4 veracity labels I am descarding all samples from the dataset that have label as -1. I also notices that samples that have -1 labels usually have one of the claims/explanation missing

## Approach:
The problem can be apporached in multiple ways as follows:
    1. Since we have the explanation for why a claim was given a particular label, we can use a sentiment analysis based approach where the expalnation can be encoded using a transformer model followed by the FFN for classifying the explanation into 4 categories
    2. The second approach is to use both the claims and the explanation jointly in a transformer model. In this way the model can learn veracity of the claim jointly with the help of the explanation.

I have implemented these two approaches using distil-bert transformer model from huggingface. The metric used for evaluating the performance of the model is accuracy. We might also want to look at the f1 scores for each class if we are interested in any particular labels. Both these apporaches give a test set and val set accuracy of around 70%

Another approach that I could think of here was to use text in addition to claims and explanation as input to the transformer. Although this looks like a overkill as the explanation was derived from the text, it does seem like a worthy try. But due to the relatively lenghty nature of the text, I was not able to experiment with this due to asoiciated GPU resource constraints.

Note: Model was trained on a single node GPU at Ohio Super Computer 

## Usage:
### 1. Environement Setup

The Transformer_Text_Classification.py is the main file. 