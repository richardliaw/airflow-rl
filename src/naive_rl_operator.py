from sagemaker.tensorflow import TensorFlow

# def RLOperator(
#         entry_point="train-%s.py" % roboschool_problem,
#         source_dir='src',
#         dependencies=["common/sagemaker_rl"],
#         image_name=gpu_image_name,
#         role=role,
#         train_instance_type=primary_cluster_instance_type,
#         secondary_cluster_instance_type=secondary_cluster_instance_type,
#         train_instance_count=primary_cluster_instance_count,
#         secondary_instance_count=secondary_cluster_instance_count,
#         output_path=s3_output_path,
#         base_job_name=job_name_prefix,
#         metric_definitions=metric_definitions,
#         train_max_run=int(3600 * .5), # Maximum runtime in seconds
#         hyperparameters={
#             "s3_prefix": s3_prefix, # Important for syncing
#             "s3_bucket": s3_bucket, # Important for syncing
#             "aws_region": boto3.Session().region_name, # Important for S3 connection
#             "rl.training.config.num_workers": total_cpus,
#             "rl.training.config.train_batch_size": 20000,
#             "rl.training.config.num_gpus": total_gpus,
#         },
#         subnets=default_subnets, # Required for VPC mode
#         security_group_ids=default_security_groups # Required for VPC mode
#         ):
#     pass
def rl_train():
    metric_definitions = RLEstimator.default_metric_definitions(RLToolkit.RAY)

    estimator = RLEstimator(entry_point="train-%s.py" % roboschool_problem,
                            source_dir='src',
                            dependencies=["common/sagemaker_rl"],
                            image_name=custom_image_name,
                            role=role,
                            train_instance_type=instance_type,
                            train_instance_count=train_instance_count,
                            output_path=s3_output_path,
                            base_job_name=job_name_prefix,
                            metric_definitions=metric_definitions,
                            hyperparameters={
                              # Attention scientists!  You can override any Ray algorithm parameter here:

                                # 3 m4.2xl with 8 cores each. We have to leave 1 core for ray scheduler.
                                # Don't forget to change this on the basis of instance type.
                                "rl.training.config.num_workers": (8 * train_instance_count) - 1

                              #"rl.training.config.horizon": 5000,
                              #"rl.training.config.num_sgd_iter": 10,
                            }
                        )

    estimator.fit(wait=local_mode)
    job_name = estimator.latest_training_job.job_name
    print("Training job: %s" % job_name)
    return job_name


# callable for SageMaker training in TensorFlow
def train(data, **context):
    estimator = TensorFlow(entry_point='tf_train.py',
                           role='sagemaker-role',
                           framework_version='1.11.0',
                           training_steps=1000,
                           evaluation_steps=100,
                           train_instance_count=2,
                           train_instance_type='ml.p2.xlarge')
    estimator.fit(data)
    return estimator.latest_training_job.job_name

# # callable for SageMaker batch transform
# def transform(data, **context):
#     training_job = context['ti'].xcom_pull(task_ids='training')
#     estimator = TensorFlow.attach(training_job)
#     transformer = estimator.transformer(instance_count=1, instance_type='ml.c4.xlarge')
#     transformer.transform(data, content_type='text/csv')

import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner': 'airflow',
    'start_date': airflow.utils.dates.days_ago(2),
    'provide_context': True
}

dag = DAG('rl_example', default_args=default_args,
          schedule_interval='@once')

train_op = PythonOperator(
    task_id='training',
    python_callable=train,
    op_args=[training_data_s3_uri],
    provide_context=True,
    dag=dag)
