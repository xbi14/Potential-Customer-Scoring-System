from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.sensors.time_delta_sensor import TimeDeltaSensor
from airflow.hooks.postgres_hook import PostgresHook
from datetime import datetime, timedelta
import boto3
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'catchup':False,
}

dag = DAG(
    's3_postgres',
    default_args=default_args,
    description='Download file from S3 and insert into PostgreSQL',
    schedule_interval='@daily',
    start_date=datetime(2024, 5, 5),
)

# S3 and DB credentials
AWS_ACCESS_KEY = 'AKIA4WHQVK4DLLWIWANL'
AWS_SECRET_KEY = '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'



S3_FILE = 'data_1000.csv'

DB_CONN_ID = 'postgres_default'
DB_TABLE = 'potential_customers'

def download_file_from_s3():
    s3 = boto3.resource('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    try:
        for bucket in s3.buckets.all():
            bucket_name = bucket.name
            s3_bucket = s3.Bucket(bucket_name)
            for s3_file in s3_bucket.objects.all():
                if s3_file.key == S3_FILE:
                    file_path = f"/tmp/{S3_FILE}"
                    s3.Bucket(bucket_name).download_file(s3_file.key, file_path)
                    return file_path
    except Exception as e:
        print(f"Failed to find the bucket and corresponding files: {str(e)}")
        return None

def create_table_postgres():
    postgres_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS potential_customers (
        UserID TEXT,
        basket_icon_click INTEGER,
        basket_add_list INTEGER,
        basket_add_detail INTEGER,
        sort_by INTEGER,
        image_picker INTEGER,
        account_page_click INTEGER,
        promo_banner_click INTEGER,
        detail_wishlist_add INTEGER,
        list_size_dropdown INTEGER,
        closed_minibasket_click INTEGER,
        checked_delivery_detail INTEGER,
        checked_returns_detail INTEGER,
        sign_in INTEGER,
        saw_checkout INTEGER,
        saw_sizecharts INTEGER,
        saw_delivery INTEGER,
        saw_account_upgrade INTEGER,
        saw_homepage INTEGER,
        device_mobile INTEGER,
        device_computer INTEGER,
        device_tablet INTEGER,
        returning_user INTEGER,
        loc_uk INTEGER,
        ordered INTEGER,
        created_at TIMESTAMP
    );
    '''
    postgres_hook.run(create_table_query)
    print('Create table successful')

def insert_data_into_postgres(file_path):
    df = pd.read_csv(file_path)
    df['created_at'] = datetime.now()
    records = df.to_dict(orient='records')

    postgres_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    conn = postgres_hook.get_conn()
    cursor = conn.cursor()

    for record in records:
        cols = ', '.join(record.keys())
        values = ', '.join(['%s'] * len(record))
        insert_query = f"INSERT INTO {DB_TABLE} ({cols}) VALUES ({values})"
        cursor.execute(insert_query, list(record.values()))

    conn.commit()
    cursor.close()
    conn.close()
    print("Data insertion complete")

def fetch_recent_data_to_csv():
    import os
    postgres_hook = PostgresHook(postgres_conn_id=DB_CONN_ID)
    conn = postgres_hook.get_conn()
    cursor = conn.cursor()

    query = f"SELECT * FROM {DB_TABLE} WHERE created_at >= NOW() - INTERVAL '1 day';;"

    cursor.execute(query)
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=colnames)
    # Create the directory if it doesn't exist
    output_dir = '/opt/airflow/data'
    os.makedirs(output_dir, exist_ok=True)

    # Save the CSV file
    output_file = os.path.join(output_dir, 'recent_data.csv')
    df.to_csv(output_file, index=False)

    cursor.close()
    conn.close()
    print("Data fetch complete")

def download_and_insert_data():
    data = download_file_from_s3()
   
    insert_data_into_postgres(data)

with dag:
    download_and_insert_task = PythonOperator(
        task_id='download_and_insert_data',
        python_callable=download_and_insert_data,
        op_args=[],
        provide_context=True,
    )

    create_table_task = PythonOperator(
        task_id='create_postgres_table',
        python_callable=create_table_postgres,
    )

    fetch_recent_data_task = PythonOperator(
        task_id='fetch_recent_data',
        python_callable=fetch_recent_data_to_csv,
    )

    wait_until_6pm = TimeDeltaSensor(
        task_id='wait_until_6pm',
        delta=timedelta(hours=18),  # Chờ đến 18h
        mode='reschedule'
    )

    create_table_task >> download_and_insert_task >> wait_until_6pm >> fetch_recent_data_task 
