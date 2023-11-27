import sagemaker

sagemaker_session = sagemaker.session.Session()
bucket = sagemaker_session.default_bucket()
role=sagemaker.get_execution_role()

prefix="myprefix"

estimator = sagemaker.estimator.PyTorch(
    framework_version="1.11.0",
    py_version="py38",
    role,
    entry_point="train.py", #the entry point for your training code
    source_dir="src", #path to your local folder containing your training code
    instance_count=1,
    instance_type="ml.g5.2xlarge",
    hyperparameters={"epochs": 3}
)

train_config = sagemaker.TrainingInput('s3://{0}/{1}/train/'.format(bucket, prefix), content_type='text/csv')
val_config = sagemaker.TrainingInput('s3://{0}/{1}/val/'.format(bucket, prefix), content_type='text/csv')

estimator.fit({'train': train_config, 'validation': val_config })