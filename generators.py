import numpy as np
from psycopg2 import connect, ProgrammingError


def choice(filenames, curr_filenames):
    filename = filenames[np.random.randint(0, len(filenames))]
    k = 0
    while filename in curr_filenames and k < 50:
        filename = filenames[np.random.randint(0, len(filenames))]
        k += 1
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

        try:
            sql = f'SELECT x, main_target, state, filename FROM mammologic_dataset WHERE filename IN ({str(curr_filenames)[1:-1]})'
            cursor.execute(sql)
        except Exception as er:
            print(er, batch_size, curr_filenames)
            batch_size = 32
            continue

        xy_by_filename = {filename: {'y': None, 'x0': None, 'x3': None, 'x6': None, 'x9': None, 'rev_x0': None,
                                     'rev_x3': None, 'rev_x6': None, 'rev_x9': None} for filename in curr_filenames}

        try:
            c = False
            for row in cursor:
                x, y, state, filename = row
                try:
                    if x is None or y is None:
                        raise KeyError
                    xy_by_filename[filename][state] = x
                    xy_by_filename[filename]['y'] = y
                except KeyError:
                    c = True
        except ProgrammingError as er:
            print(er)
            print(er, batch_size, curr_filenames)
            batch_size = 32
            continue

        if c:
            for filename in list(xy_by_filename.keys()):
                xy = xy_by_filename[filename]
                b = False
                for key in xy.keys():
                    b = xy[key] is None or b

                if b:
                    xy_by_filename.pop(filename)

            if len(xy_by_filename.keys()) == 0:
                print(f'len(xy_by_filename.keys()) {len(xy_by_filename.keys())}')
                continue

        xs = []
        ys = []
        bs = len(xy_by_filename.keys())
        print('keys', bs, xy_by_filename.keys())
        for filename in list(xy_by_filename.keys()):
            xy = xy_by_filename[filename]

            c = False
            for key in xy.keys():
                c = xy[key] is None or c

            if c:
                bs -= 1
                continue

            y = np.zeros(2, dtype=dtype)
            if xy.pop('y') is True:
                y[1] = 1.0  # sick
            else:
                y[0] = 1.0  # healthy

            try:
                x = np.array(xy.pop('x0'), dtype=dtype)

                for key in xy.keys():
                    xy[key] = np.array(xy[key], dtype=dtype)

                print([(key, xy[key].shape) for key in xy.keys()])
                for key in order[1:]:
                    next_x = xy.pop(key)
                    x = np.concatenate((x, next_x), axis=0)
            except ValueError as er:
                bs -= 1
                print(er, bs)
                continue

            xs.append(x)
            ys.append(y)

        xs = np.array(xs, dtype=dtype)
        if len(xs.shape) < 4:
            print(f'{xs.shape} xs.shape')
            continue

        xs = np.expand_dims(xs / np.amax(xs), axis=5)
        xs = xs - np.mean(xs)

        xs = [xs[:, i, :, :, :, :] for i in range(18 * 8)]
        y = np.array(ys, dtype=dtype)
        ys = [np.zeros((bs, 1), dtype=dtype),
              np.zeros((bs, 1), dtype=dtype),
              np.zeros((bs, 1), dtype=dtype)]
        for _ in range((8 + 1) * 2):
            ys.append(y)

        xs.append(y)
        yield xs, ys


#  xs, ys = next(base_generator(2))

