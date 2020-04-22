import numpy as np
from psycopg2 import connect


def choice(filenames, curr_filenames):
    filename = filenames[np.random.randint(0, len(filenames))]
    while filename in curr_filenames:
        filename = filenames[np.random.randint(0, len(filenames))]
    return filename


conn = connect(host='161.35.19.170', database='renova_datasets', user='postgres', password='postgres')
cursor = conn.cursor()
mfilenames = []

cursor.execute('SELECT filename FROM mammologic_dataset')

for row in cursor:
    mfilenames.append(row[0])

mfilenames = list(set(mfilenames))

order = ['x0', 'x3', 'x6', 'x9', 'rev_x0', 'rev_x3', 'rev_x6', 'rev_x9']


def base_generator(batch_size, is_val=False, dtype=np.float32):
    if is_val:
        filenames = mfilenames[::3]
    else:
        filenames = list(set(mfilenames) - set(mfilenames[::3]))

    while True:
        curr_filenames = set()

        for _ in range(batch_size):
            filename = choice(filenames, curr_filenames)
            curr_filenames.add(filename)

        sql = f'SELECT x, main_target, state, filename FROM mammologic_dataset WHERE filename IN ({str(curr_filenames)[1:-1]})'
        cursor.execute(sql)

        xy_by_filename = {filename: {'y': None, 'x0': None, 'x3': None, 'x6': None, 'x9': None, 'rev_x0': None,
                                     'rev_x3': None, 'rev_x6': None, 'rev_x9': None} for filename in curr_filenames}

        for row in cursor:
            x, y, state, filename = row
            xy_by_filename[filename][state] = x
            xy_by_filename[filename]['y'] = y

        xs = []
        ys = []

        for filename in xy_by_filename.keys():
            xy = xy_by_filename[filename]
            # c = False
            # for key in xy.keys():
            #     c = xy[key] is None or c
            #
            # if c:
            #     continue
            y = np.zeros(2, dtype=dtype)
            if xy.pop('y') is True:
                y[1] = 1.0  # sick
            else:
                y[0] = 1.0  # healthy

            x = np.array(xy.pop('x0'), dtype=dtype)

            for key in order[1:]:
                x = np.concatenate((x, xy.pop(key)), axis=0)

            xs.append(x)
            ys.append(y)

        xs = np.array(xs, dtype=dtype)
        xs = [xs[:, i, :, :, :] for i in range(18 * 8)]
        #ys = [np.array(ys), np.zeros(1, dtype=dtype)]
        ys = [np.zeros(1, dtype=dtype), np.zeros(1, dtype=dtype), np.zeros(1, dtype=dtype), np.array(ys, dtype=dtype)]
        xs.append(ys)

        yield xs, ys


# xs, ys = next(base_generator(2))

