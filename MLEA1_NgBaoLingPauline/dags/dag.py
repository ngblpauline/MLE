from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")

    bronze_label = BashOperator(
        task_id='bronze_label_store',
        bash_command=(
            'cd /opt/airflow/scripts/utils && '
            'python3 data_processing_bronze_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_label = BashOperator(
        task_id='silver_label_store',
        bash_command=(
            'cd /opt/airflow/scripts/utils && '
            'python3 data_processing_silver_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    gold_label = BashOperator(
        task_id='gold_label_store',
        bash_command=(
            'cd /opt/airflow/scripts/utils && '
            'python3 data_processing_gold_table.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    dep_check_source_label_data >> bronze_label >> silver_label >> gold_label
 
 
    # --- feature store ---
    dep_check_bronze_feat = DummyOperator(task_id="dep_check_source_data_bronze_1")

    bronze_feat = BashOperator(
        task_id='bronze_table_feature',
        bash_command=(
            'python3 /opt/airflow/scripts/utils/data_processing_bronze_table_feature.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    silver_feat = BashOperator(
        task_id='silver_table_feature',
        bash_command=(
            'python3 /opt/airflow/scripts/utils/data_processing_silver_table_feature.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    gold_feat = BashOperator(
        task_id='gold_table_feature',
        bash_command=(
            'python3 /opt/airflow/scripts/utils/data_processing_gold_table_feature.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    dep_check_bronze_feat >> bronze_feat >> silver_feat >> gold_feat

    # MERGE LABEL + FEATURE INTO final_merged.csv
    merge_data = BashOperator(
        task_id='merge_labels_and_features',
        bash_command=(
            'python3 /opt/airflow/scripts/merge.py '
            '--snapshotdate "{{ ds }}" '
            '--label_path "/opt/airflow/scripts/datamart/gold/label_store/" '
            '--feature_path "/opt/airflow/scripts/datamart/gold/feature/" '
            '--output_path "/opt/airflow/scripts/data/final_merged.csv"'
        )
    )

    [gold_label, gold_feat] >> merge_data


    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_1_inference = BashOperator(
        task_id="model_1_inference",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_inference.py "
            "--modelname xgboostv1.json "
            "--inputpath data/final_merged_1.csv"
        ),
    )

    merge_data >> model_inference_start >> model_1_inference 



    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_1_monitor = BashOperator(
    task_id="model_1_monitor",
    bash_command='python /opt/airflow/scripts/monitoring.py')


    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_1_inference >> model_monitor_start
    model_monitor_start >> model_1_monitor >> model_monitor_completed


    # --- model auto training ---

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_1_automl = DummyOperator(task_id="model_1_automl")

    model_2_automl = DummyOperator(task_id="model_2_automl")

    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    gold_feat >> model_automl_start
    gold_label >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed