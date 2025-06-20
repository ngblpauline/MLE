from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email': ['hydrogen.sulphate@gmail.com'],
    'email_on_failure': True,     
    'email_on_retry': False,
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 6, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    label_store = BashOperator(
        task_id='run_label_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 main_label.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    label_store_completed = DummyOperator(task_id="label_store_completed")

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> label_store >> label_store_completed
 
 
    # --- feature store ---
    dep_check_source_feature_data = DummyOperator(task_id="dep_check_source_feature_data")

    feature_store = BashOperator(
        task_id='feature_store',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 main.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_feature_data >> feature_store >> feature_store_completed

    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "xgboostv1.json"'
        ),
    )

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_inference >> model_inference_completed


    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_monitor.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "xgboostv1.json"'
        ),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_monitor >> model_monitor_completed