# Sentiment Analysis using XGBoost in Amazon SageMaker 

This mini-project uses Amazon's SageMaker service to construct random tree models with the XGBoost package and predict the sentiment of a movie review as positive or negative using the [IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/). The trained machine learning models are then deployed to a production environment using [Amazon SageMaker](https://aws.amazon.com/sagemaker). 

# Install

The exercises require Python 3.6 and the following Python libraries installed   
  - [numpy](https://numpy.org/)

# Run

To open the .ipynb files in your browser and look at the output of the completed cells, use the following command in your terminal after changing the working directory to the coursework directory `sentiment_analysis_xgboost`:
```
jupyter notebook <file_name>.ipynb
```

To run and execute all the cells from scratch, you need to create a Jupyter notebook instance in the Amazon Sagemaker service, configure the lifecycle of the notebook, and attache the [Github repository](https://github.com/wchowdhu/udacity-ml-engineer-nanodegree.git) to the instance. Once set up, you can run the cells to upload the data and save any model artifacts of the trained model in [AWS S3](https://aws.amazon.com/s3/) service.  



