from __future__ import print_function
import json
# import requests
# from datetime import datetime

# airflow operators
import airflow
from airflow.models import DAG
# from airflow.utils.trigger_rule import TriggerRule
# from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
# from airflow.operators.python_operator import PythonOperator

# airflow sagemaker operators
from airflow.contrib.operators.sagemaker_training_operator \
    import SageMakerTrainingOperator
# from airflow.contrib.operators.sagemaker_tuning_operator \
#     import SageMakerTuningOperator
# from airflow.contrib.operators.sagemaker_transform_operator \
#     import SageMakerTransformOperator
from airflow.contrib.hooks.aws_hook import AwsHook

# sagemaker sdk
# import boto3
import sagemaker
# from sagemaker.amazon.amazon_estimator import get_image_uri
# from sagemaker.estimator import Estimator
# from sagemaker.tuner import HyperparameterTuner

# airflow sagemaker configuration
from sagemaker.workflow.airflow import training_config
# from sagemaker.workflow.airflow import tuning_config
# from sagemaker.workflow.airflow import transform_config_from_estimator

# ml workflow specific
# from pipeline import prepare, preprocess
# import config as cfg

roboschool_problem = 'reacher'

# =============================================================================
# functions
# =============================================================================
import boto3
import sys
import os
import glob
import re
import subprocess
# from IPython.display import HTML, Markdown
import time
# from time import gmtime, strftime
from smrl_common.misc import get_execution_role, wait_for_s3_object
from smrl_common.docker_utils import build_and_push_docker_image
from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
from smrl_common.markdown_helper import generate_help_for_s3_endpoint_permissions, create_s3_endpoint_manually

def get_sagemaker_role_arn(role_name, region_name):
    iam = boto3.client('iam', region_name=region_name)
    response = iam.get_role(RoleName=role_name)
    return response["Role"]["Arn"]

# =============================================================================
# setting up training, tuning and transform configuration
# =============================================================================

sess = sagemaker.session.Session()
# set configuration for tasks
hook = AwsHook(aws_conn_id='airflow-sagemaker')
region = "us-east-1"
# sess = hook.get_session(region_name=region)
role = get_execution_role()# get_sagemaker_role_arn(
    # "AirflowSageMakerExecutionRole",
    # sess.region_name)
# container = get_image_uri(sess.region_name, 'factorization-machines')
# hpo_enabled = is_hpo_enabled()

# print(sess.default_bucket())
# print(hook.get_session())
# import ipdb; ipdb.set_trace()
cpu_or_gpu = "cpu" #'gpu' if instance_type.startswith('ml.p') else 'cpu'
repository_short_name = "sagemaker-roboschool-ray-%s" % cpu_or_gpu
docker_build_args = {
    'CPU_OR_GPU': cpu_or_gpu,
    'AWS_REGION': boto3.Session().region_name,
}
custom_image_name = build_and_push_docker_image(repository_short_name, build_args=docker_build_args)
print("Using ECR image %s" % custom_image_name)

metric_definitions = RLEstimator.default_metric_definitions(RLToolkit.RAY)

estimator = RLEstimator(
    entry_point="train-%s.py" % roboschool_problem,
    source_dir='src',
    dependencies=["common/sagemaker_rl"],
    image_name=custom_image_name,
    role=role,
    train_instance_type="ml.c5.4xlarge",
    train_instance_count=2,
    output_path='s3://{}/'.format(sess.default_bucket()),
    base_job_name="rl-test",
    metric_definitions=metric_definitions,
    hyperparameters={
      # Attention scientists!  You can override any Ray algorithm parameter here:

        # 3 m4.2xl with 8 cores each. We have to leave 1 core for ray scheduler.
        # Don't forget to change this on the basis of instance type.
        "rl.training.config.num_workers": (8 * 2) - 1

      #"rl.training.config.horizon": 5000,
      #"rl.training.config.num_sgd_iter": 10,
    }
)

# estimator.fit(wait=local_mode)
# job_name = estimator.latest_training_job.job_name
# print("Training job: %s" % job_name)

# train_config specifies SageMaker training configuration
train_config = training_config(
    estimator=estimator,
    inputs="s3://sagemaker-us-east-1-450145409201/sagemaker/DEMO-pytorch-mnist")  # MOCK
# inputs=config["train_model"]["inputs"])

# # create tuner
# fm_tuner = HyperparameterTuner(
#     estimator=fm_estimator,
#     **config["tune_model"]["tuner_config"]
# )

# # create tuning config
# tuner_config = tuning_config(
#     tuner=fm_tuner,
#     inputs=config["tune_model"]["inputs"])

# # create transform config
# transform_config = transform_config_from_estimator(
#     estimator=fm_estimator,
#     task_id="model_tuning" if hpo_enabled else "model_training",
#     task_type="tuning" if hpo_enabled else "training",
#     **config["batch_transform"]["transform_config"]
# )

# # =============================================================================
# define airflow DAG and tasks
# =============================================================================

# define airflow DAG

args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    dag_id='sagemaker-ml-pipeline',
    default_args=args,
    schedule_interval=None,
    concurrency=1,
    max_active_runs=1,
    user_defined_filters={'tojson': lambda s: json.JSONEncoder().encode(s)}
)

# set the tasks in the DAG

# dummy operator
init = DummyOperator(
    task_id='start',
    dag=dag
)

# launch sagemaker training job and wait until it completes
train_model_task = SageMakerTrainingOperator(
    task_id='model_training',
    dag=dag,
    config=train_config,
    aws_conn_id='airflow-sagemaker',
    wait_for_completion=True,
    check_interval=30
)

# # launch sagemaker hyperparameter job and wait until it completes
# tune_model_task = SageMakerTuningOperator(
#     task_id='model_tuning',
#     dag=dag,
#     config=tuner_config,
#     aws_conn_id='airflow-sagemaker',
#     wait_for_completion=True,
#     check_interval=30
# )

# # launch sagemaker batch transform job and wait until it completes
# batch_transform_task = SageMakerTransformOperator(
#     task_id='predicting',
#     dag=dag,
#     config=transform_config,
#     aws_conn_id='airflow-sagemaker',
#     wait_for_completion=True,
#     check_interval=30,
#     trigger_rule=TriggerRule.ONE_SUCCESS
# )

cleanup_task = DummyOperator(
    task_id='cleaning_up',
    dag=dag)

# set the dependencies between tasks

# init.set_downstream(preprocess_task)
# preprocess_task.set_downstream(prepare_task)
# prepare_task.set_downstream(branching)
# branching.set_downstream(tune_model_task)
# branching.set_downstream(train_model_task)
# tune_model_task.set_downstream(batch_transform_task)
# train_model_task.set_downstream(batch_transform_task)
# batch_transform_task.set_downstream(cleanup_task)
init.set_downstream(train_model_task)
train_model_task.set_downstream(cleanup_task)
