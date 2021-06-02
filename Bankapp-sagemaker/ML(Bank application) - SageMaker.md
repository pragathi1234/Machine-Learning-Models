# Importing Important Libraries
Steps To Be Followed
Importing necessary Libraries
Creating S3 bucket
Mapping train And Test Data in S3
Mapping The path of the models in S3

```python
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import image_uris
from sagemaker.session import s3_input, Session
```


```python
#bucket_name = 'mlproject' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
my_region = boto3.session.Session().region_name # set the region of the instance
print(my_region)
```

    us-east-1

s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)

```python
bucket_name='sagemaker-basicdemo'
# set an output path where the trained model will be saved
prefix = 'xgboost-as-a-built-in-algo'
output_path ='s3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)
```

    s3://sagemaker-basicdemo/xgboost-as-a-built-in-algo/output


# Downloading The Dataset And Storing in S3


```python
import pandas as pd
import urllib
try:
    urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print('Success: downloaded bank_clean.csv.')
except Exception as e:
    print('Data load error: ',e)

try:
    model_data = pd.read_csv('./bank_clean.csv',index_col=0)
    print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```

    Success: downloaded bank_clean.csv.
    Success: Data loaded into dataframe.



```python
### Train Test split

import numpy as np
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)

```

    (28831, 61) (12357, 61)



```python
### Saving Train And Test Into Buckets
## We start with Train Data
import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], 
                                                axis=1)], 
                                                axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
```


```python
# Test Data Into Buckets
pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')
```

# Building Models Xgboot- Inbuilt Algorithm


```python
# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.

xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "1.2-2")

```


```python
# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"binary:logistic",
        "num_round":50
        }

```


```python
estimator = sagemaker.estimator.Estimator(image_uri=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          train_instance_count=1, 
                                          train_instance_type='ml.m5.2xlarge', 
                                          train_volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)
```

    train_instance_count has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_instance_type has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_max_run has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_use_spot_instances has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_max_wait has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_volume_size has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.



```python
estimator.fit({'train': s3_input_train,'validation': s3_input_test})
```

    2021-06-01 07:28:26 Starting - Starting the training job...
    2021-06-01 07:28:50 Starting - Launching requested ML instancesProfilerReport-1622532506: InProgress
    ......
    2021-06-01 07:29:50 Starting - Preparing the instances for training......
    2021-06-01 07:30:50 Downloading - Downloading input data
    2021-06-01 07:30:50 Training - Downloading the training image...
    2021-06-01 07:31:24 Training - Training image download completed. Training in progress..[34mINFO:sagemaker-containers:Imported framework sagemaker_xgboost_container.training[0m
    [34mINFO:sagemaker-containers:Failed to parse hyperparameter objective value binary:logistic to Json.[0m
    [34mReturning the value itself[0m
    [34mINFO:sagemaker-containers:No GPUs detected (normal if no gpus installed)[0m
    [34mINFO:sagemaker_xgboost_container.training:Running XGBoost Sagemaker in algorithm mode[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[07:31:26] 28831x59 matrix with 1701029 entries loaded from /opt/ml/input/data/train?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Determined delimiter of CSV input is ','[0m
    [34m[07:31:26] 12357x59 matrix with 729063 entries loaded from /opt/ml/input/data/validation?format=csv&label_column=0&delimiter=,[0m
    [34mINFO:root:Single node training.[0m
    [34mINFO:root:Train matrix has 28831 rows[0m
    [34mINFO:root:Validation matrix has 12357 rows[0m
    [34m[07:31:26] WARNING: /workspace/src/learner.cc:328: [0m
    [34mParameters: { num_round } might not be used.
    
      This may not be accurate due to some parameters are only used in language bindings but
      passed down to XGBoost core.  Or some parameters are not used but slip through this
      verification. Please open an issue if you find above cases.
    
    [0m
    [34m[0]#011train-error:0.10079#011validation-error:0.10528[0m
    [34m[1]#011train-error:0.09968#011validation-error:0.10456[0m
    [34m[2]#011train-error:0.10017#011validation-error:0.10375[0m
    [34m[3]#011train-error:0.09989#011validation-error:0.10310[0m
    [34m[4]#011train-error:0.09996#011validation-error:0.10286[0m
    [34m[5]#011train-error:0.09906#011validation-error:0.10261[0m
    [34m[6]#011train-error:0.09930#011validation-error:0.10286[0m
    [34m[7]#011train-error:0.09951#011validation-error:0.10261[0m
    [34m[8]#011train-error:0.09920#011validation-error:0.10286[0m
    [34m[9]#011train-error:0.09871#011validation-error:0.10294[0m
    [34m[10]#011train-error:0.09868#011validation-error:0.10294[0m
    [34m[11]#011train-error:0.09868#011validation-error:0.10326[0m
    [34m[12]#011train-error:0.09854#011validation-error:0.10358[0m
    [34m[13]#011train-error:0.09892#011validation-error:0.10342[0m
    [34m[14]#011train-error:0.09850#011validation-error:0.10342[0m
    [34m[15]#011train-error:0.09844#011validation-error:0.10326[0m
    [34m[16]#011train-error:0.09857#011validation-error:0.10318[0m
    [34m[17]#011train-error:0.09799#011validation-error:0.10318[0m
    [34m[18]#011train-error:0.09816#011validation-error:0.10383[0m
    [34m[19]#011train-error:0.09857#011validation-error:0.10383[0m
    [34m[20]#011train-error:0.09830#011validation-error:0.10350[0m
    [34m[21]#011train-error:0.09826#011validation-error:0.10318[0m
    [34m[22]#011train-error:0.09847#011validation-error:0.10399[0m
    [34m[23]#011train-error:0.09833#011validation-error:0.10407[0m
    [34m[24]#011train-error:0.09812#011validation-error:0.10415[0m
    [34m[25]#011train-error:0.09812#011validation-error:0.10399[0m
    [34m[26]#011train-error:0.09774#011validation-error:0.10375[0m
    [34m[27]#011train-error:0.09781#011validation-error:0.10375[0m
    [34m[28]#011train-error:0.09781#011validation-error:0.10391[0m
    [34m[29]#011train-error:0.09778#011validation-error:0.10367[0m
    [34m[30]#011train-error:0.09781#011validation-error:0.10383[0m
    [34m[31]#011train-error:0.09771#011validation-error:0.10358[0m
    [34m[32]#011train-error:0.09743#011validation-error:0.10391[0m
    [34m[33]#011train-error:0.09753#011validation-error:0.10342[0m
    [34m[34]#011train-error:0.09767#011validation-error:0.10342[0m
    [34m[35]#011train-error:0.09757#011validation-error:0.10350[0m
    [34m[36]#011train-error:0.09757#011validation-error:0.10342[0m
    [34m[37]#011train-error:0.09736#011validation-error:0.10342[0m
    [34m[38]#011train-error:0.09750#011validation-error:0.10342[0m
    [34m[39]#011train-error:0.09733#011validation-error:0.10350[0m
    [34m[40]#011train-error:0.09705#011validation-error:0.10358[0m
    [34m[41]#011train-error:0.09701#011validation-error:0.10383[0m
    [34m[42]#011train-error:0.09712#011validation-error:0.10407[0m
    [34m[43]#011train-error:0.09698#011validation-error:0.10375[0m
    [34m[44]#011train-error:0.09733#011validation-error:0.10342[0m
    [34m[45]#011train-error:0.09736#011validation-error:0.10367[0m
    [34m[46]#011train-error:0.09746#011validation-error:0.10350[0m
    [34m[47]#011train-error:0.09736#011validation-error:0.10358[0m
    [34m[48]#011train-error:0.09712#011validation-error:0.10334[0m
    [34m[49]#011train-error:0.09712#011validation-error:0.10318[0m
    
    2021-06-01 07:31:50 Uploading - Uploading generated training model
    2021-06-01 07:31:50 Completed - Training job completed
    Training seconds: 67
    Billable seconds: 31
    Managed Spot Training savings: 53.7%


# Deploy Machine Learning Model As Endpoints



```python
xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

    -----------------!

# Prediction of the Test Data


```python
xgb_predictor.__dict__.keys()
```




    dict_keys(['endpoint_name', 'sagemaker_session', 'serializer', 'deserializer', '_endpoint_config_name', '_model_names', '_context'])




```python
from sagemaker.predictor import csv_serializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
#xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
```

    The csv_serializer has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.


    (12357,)



```python
predictions_array
```




    array([0.05214286, 0.05660191, 0.05096195, ..., 0.03436061, 0.02942475,
           0.03715819])




```python
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

    
    Overall Classification Rate: 89.7%
    
    Predicted      No Purchase    Purchase
    Observed
    No Purchase    91% (10785)    34% (151)
    Purchase        9% (1124)     66% (297) 
    



```python
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```
