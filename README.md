# ml-webapp

## Overview

This will hold a webapp that serves various ML models to the web.

## Models

* The pet classifier model is trained in (this notebook)[https://github.com/robwil/fastai-deeplearning/blob/master/Pet_Classification.ipynb] using FastAI's wrapping of PyTorch.
* The flower classifier model is trained in (this notebook)[https://github.com/robwil/fastai-deeplearning/blob/master/Flower_Classification.ipynb] using FastAI's wrapping of PyTorch.

The PyTorch models are exported as an ONNX model because they have less runtime dependencies than PyTorch and/or FastAI (1.5GB docker image vs. 4.5GB).

## Deployment

Deploying using Google Cloud Run for simplicity / cost.

```bash
gcloud builds submit --tag gcr.io/robwil-io/ml-webapp
gcloud beta run deploy --image gcr.io/robwil-io/ml-webapp --platform managed --memory 2G
```