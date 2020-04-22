from psycopg2 import connect

conn = connect(dbname='renova_datasets', user='postgres', password='postgres')
cur = conn.cursor()

tasks = """
CREATE TABLE tasks (id SERIAL, skill varchar(20), arguments json, attempts smallint, 
result json, priority smallint, unq text, target varchar(30),
created timestamp, taken timestamp, completed timestamp, viewed timestamp, access_after timestamp)
"""

cur.execute(tasks)

mammologic_dataset = """
CREATE TABLE mammologic_dataset (id SERIAL PRIMARY KEY, filename varchar(80), x int[][][][], state varchar(6), 
number_of_dir int, original_dir varchar(80), patient_name varchar(40), is_left boolean, main_target boolean)
"""

cur.execute(mammologic_dataset)

accuracy_by_model = """
CREATE TABLE accuracy_by_model (id int REFERENCES mammologic_dataset (id), model_name varchar(250), cancer_points int)
"""

cur.execute(accuracy_by_model)

conn.commit()
cur.close()
conn.close()
