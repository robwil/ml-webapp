# ml-webapp

## Overview

This will hold a webapp that serves various ML models to the web.

## TODO

Right now the Dockerfile pulls in `fastai` (and doing so requires using the full Starlette base image, instead of the alpine one), but that has a crazy amount of dependencies. This leads to a Docker image of 4GB :( Ideally, I want to figure out how to run these models in a lighter weight form, but it's fine for now in this POC.

## Deployment

Deploying using Google Cloud Run for simplicity / cost.

```bash
gcloud builds submit --tag gcr.io/robwil-io/ml-webapp
gcloud beta run deploy --image gcr.io/robwil-io/ml-webapp --platform managed --memory 2G
```