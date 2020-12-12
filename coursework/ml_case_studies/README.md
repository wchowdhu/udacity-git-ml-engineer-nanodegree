# Machine Learning Case Studies 

Case studies are in-depth examinations of specific, real-world tasks. The repo contains case studies focusing on different problems and how they can be solved using various machine learning strategies. 

The case studies are as follows:
  - [Population Segmentation using SageMaker](https://github.com/wchowdhu/udacity-git-ml-engineer-nanodegree/blob/master/coursework/ml_case_studies/population_segmentation/Pop_Segmentation_Exercise.ipynb) - This case study looks at a portion of [US census data](https://www.census.gov/data.html) and, using a combination of unsupervised learning methods, extracts meaningful components from that data and groups regions by similar census-recorded characteristics. The end result will be groupings that are used to inform things like localized marketing campaigns and voter campaign strategies.
  
  - [Detecting Credit Card Fraud](https://github.com/wchowdhu/udacity-git-ml-engineer-nanodegree/blob/master/coursework/ml_case_studies/fraud_detection/Fraud_Detection_Exercise.ipynb) - This case demonstrates how to use supervised learning techniques, specifically SageMaker’s LinearLearner, for fraud detection. The payment transaction dataset is unbalanced, with many more examples of valid transactions vs. fraudulent, and so methods for compensating for this imbalance are investigated.
  
  - [Custom Models: Non-Linear Classification](https://github.com/wchowdhu/udacity-git-ml-engineer-nanodegree/tree/master/coursework/ml_case_studies/moon_classification) - This case study looks into cases where classes of data are not separable by a linear line. So a custom, PyTorch neural network is trained and deployed for classifying the data.
  
  - [Time-Series Forecasting](https://github.com/wchowdhu/udacity-git-ml-engineer-nanodegree/tree/master/coursework/ml_case_studies/time_series_forecasting) - This case demonstrates how to train SageMaker's DeepAR model for forecasting predictions over time.

# Install

The exercises require Python 3.6 and the following Python libraries installed   
  - [numpy](https://numpy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [scikit-learn](https://scikit-learn.org/stable/)

# Run

To open the .ipynb files in your browser and look at the output of the completed cells, use the following command in your terminal after changing the working directory to the coursework directory with the corresponding folder name of a case study `ml_case_studies/<folder_name>`:
```
jupyter notebook <file_name>.ipynb
```

To run and execute all the cells from scratch, you need to create a Jupyter notebook instance in the Amazon Sagemaker service, configure the lifecycle of the notebook, and attache the [Github repository](https://github.com/wchowdhu/udacity-ml-engineer-nanodegree.git) to the instance. Once set up, you can run the cells to upload the data and save any model artifacts of the trained model in [AWS S3](https://aws.amazon.com/s3/) service 



