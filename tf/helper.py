


import datetime
import dateutil.tz
def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y%m%d_%H%M%S')


import scipy.misc
def save_images(images, height, width, n_row, n_col,
        cmin=0.0, cmax=1.0, dir="./", prefix = ''):
    images = images.reshape((n_row, n_col, height, width))
    images = images.transpose(1, 2, 0, 3)
    images = images.reshape((height * n_row, width * n_col))

    filename = prefix + '%s.jpg' % (get_timestamp())
    scipy.misc.toimage(images, cmin=cmin, cmax=cmax).save(os.path.join(dir, filename))

import os
def check_and_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
