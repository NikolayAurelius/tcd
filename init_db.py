from psycopg2 import connect

conn = connect(dbname='renova_datasets', user='postgres', password='postgres')
cur = conn.cursor()


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


'{' \
'   {' \
        '{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}' \
'   }, ' \
    '{' \
'       {{9, 10}, {11, 12}}, {{13, 14}, {15, 16}}' \
'   }' \
'}'
