from psycopg2 import connect
import os
import numpy as np

conn = connect(host='161.35.19.170', database='renova_datasets', user='postgres', password='postgres')
cursor = conn.cursor()

sql = '''SELECT DISTINCT filename FROM mammologic_dataset'''
cursor.execute(sql)

lst = []
for row in cursor:
    lst.append(row[0])

print(len(lst))
lst = set(lst)
print(len(lst))

raws = os.listdir('raw_datasets')
not_in_db = []
for raw in raws:
    for diagnose in ['sick', 'healthy']:
        not_in_db.extend(os.listdir(f'raw_datasets/{raw}/{diagnose}'))

print(len(not_in_db))
not_in_db = set(not_in_db)
print(len(not_in_db))

not_in_db = not_in_db - lst

print(len(not_in_db))

paths_to_not_in_db = []
for raw in raws:
    for diagnose in ['sick', 'healthy']:
        filenames = os.listdir(f'raw_datasets/{raw}/{diagnose}')
        for filename in filenames:
            if filename in not_in_db:
                paths_to_not_in_db.append(f'raw_datasets/{raw}/{diagnose}/{filename}')
                not_in_db.remove(filename)


def get_I():
    I = np.zeros((18, 18), dtype=np.int32) - 1000
    a = 1
    for i in range(6, 18 - 6):
        I[0, i] = a
        a += 1

    for i in range(4, 18 - 4):
        I[1, i] = a
        a += 1

    for i in range(3, 18 - 3):
        I[2, i] = a
        a += 1

    for i in range(2, 18 - 2):
        I[3, i] = a
        a += 1

    for j in range(2):
        for i in range(1, 18 - 1):
            I[4 + j, i] = a
            a += 1

    for j in range(6):
        for i in range(18):
            I[6 + j, i] = a
            a += 1

    for j in range(2):
        for i in range(1, 18 - 1):
            I[12 + j, i] = a
            a += 1

    for i in range(2, 18 - 2):
        I[14, i] = a
        a += 1

    for i in range(3, 18 - 3):
        I[15, i] = a
        a += 1

    for i in range(4, 18 - 4):
        I[16, i] = a
        a += 1

    for i in range(6, 18 - 6):
        I[17, i] = a
        a += 1
    return I


I = get_I()

rev_I = I[:, ::-1]

I0 = I
rev_I0 = rev_I

I3 = np.rot90(I, k=3, axes=(0, 1))
rev_I3 = np.rot90(rev_I, k=3, axes=(0, 1))

I6 = np.rot90(I, k=2, axes=(0, 1))
rev_I6 = np.rot90(rev_I, k=2, axes=(0, 1))

I9 = np.rot90(I, k=1, axes=(0, 1))
rev_I9 = np.rot90(rev_I, k=1, axes=(0, 1))


def file_to_x(path):
    with open(path, encoding='cp1251') as f:
        need_check = True
        lst = []
        for line in f.readlines():
            if need_check and line.count('0;') != 0:
                need_check = False
            elif not need_check:
                pass
            else:
                continue

            one_x = np.zeros((18, 18))

            line = line[:-2].split(';')
            line = list(map(int, line))

            for i in range(18):
                for j in range(18):
                    one_x[i, j] = line[i * 18 + j]

            lst.append(one_x)
        x0 = np.zeros((18, 18, 18, 18))
        rev_x0 = np.zeros((18, 18, 18, 18))

        x3 = np.zeros((18, 18, 18, 18))
        rev_x3 = np.zeros((18, 18, 18, 18))

        x6 = np.zeros((18, 18, 18, 18))
        rev_x6 = np.zeros((18, 18, 18, 18))

        x9 = np.zeros((18, 18, 18, 18))
        rev_x9 = np.zeros((18, 18, 18, 18))

        for i in range(18):
            for j in range(18):
                if I0[i, j] != -1000:
                    x0[i, j] = lst[I[i, j] - 1]

                if I3[i, j] != -1000:
                    x3[i, j] = lst[I3[i, j] - 1]

                if I6[i, j] != -1000:
                    x6[i, j] = lst[I6[i, j] - 1]

                if I9[i, j] != -1000:
                    x9[i, j] = lst[I9[i, j] - 1]

                if rev_I0[i, j] != -1000:
                    rev_x0[i, j] = lst[rev_I0[i, j] - 1][:, ::-1]

                if rev_I3[i, j] != -1000:
                    rev_x3[i, j] = lst[rev_I3[i, j] - 1][:, ::-1]

                if rev_I6[i, j] != -1000:
                    rev_x6[i, j] = lst[rev_I6[i, j] - 1][:, ::-1]

                if rev_I9[i, j] != -1000:
                    rev_x9[i, j] = lst[rev_I9[i, j] - 1][:, ::-1]

    return {'x0': x0, 'x3': x3, 'x6': x6, 'x9': x9, 'rev_x0': rev_x0, 'rev_x3': rev_x3, 'rev_x6': rev_x6,
            'rev_x9': rev_x9}


print(file_to_x(paths_to_not_in_db[0])['x0'][6:12, 6:12, 6:12, 6:12])

import pandas

df = pandas.DataFrame(columns=['big_x', 'main_target'])
i = 0
for elem in paths_to_not_in_db:
    _, _, diagnose, filename = elem.split('/')
    dct_x = file_to_x(elem)
    x = dct_x.pop('x0')

    for key in list(dct_x.keys()):
        x = np.concatenate((x, dct_x.pop(key)), axis=0)

    print(x.shape)

    print(filename)
    break
