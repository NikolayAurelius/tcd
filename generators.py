import numpy as np
from psycopg2 import connect


def choice(filenames, curr_filenames):
    filename = filenames[np.random.randint(0, len(filenames))]
    while filename in curr_filenames:
        filename = filenames[np.random.randint(0, len(filenames))]
    return filename


def base_generator(batch_size, is_val=False):
    conn = connect(host='161.35.19.170', database='renova_datasets', user='postgres', password='postgres')
    cursor = conn.cursor()
    filenames = []

    cursor.execute('SELECT filename FROM mammologic_dataset')

    for row in cursor:
        filenames.append(row[0])

    filenames = list(set(filenames))
    while True:
        curr_filenames = set()

        for _ in range(batch_size):
            filename = choice(filenames, curr_filenames)
            curr_filenames.add(filename)
        curr_filenames = f'({str(curr_filenames)[1:-1]})'
        sql = f'SELECT x, main_target, state, filename FROM mammologic_dataset WHERE filename IN {curr_filenames}'
        cursor.execute(sql)

        for row in cursor:
            x, y, state, filename = row
            #print(x)
            print(y)
            print(state)
            print(filename)

        break
base_generator(1)
