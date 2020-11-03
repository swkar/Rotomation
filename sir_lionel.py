
import os, sys
import json
import random
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.data.experimental import AUTOTUNE


class SirLionel:
    target_dir = 'WACV_dataset_sir_lionel'

    def __init__(self, target_dir='WACV_dataset_sir_lionel'):
        SirLionel.target_dir = target_dir

    shot_frame_dct = {
        'shot0': 321,
        'shot1': 267,
        'shot2': 327,
        'shot3': 398,
        'shot4': 125,
        'CG':    936,
        'test':  213
    }

    raw_bezier_names = ('left_sideburn', 'left_eye',  'left_eyebrow',
                        'right_eyebrow', 'right_eye', 'right_sideburn',
                        'left_seam', 'nose_seam', 'right_seam')

    revec_bezier_names = ('left_sideburn',  'left_eye',  'left_eyebrow',
                          'right_sideburn', 'right_eye', 'right_eyebrow')

    test_bezier_names = ('left_eye',  'left_eyebrow',
                         'right_eye', 'right_eyebrow')

    points_name = 'points'

    '''
     WACV_dataset_sir_lionel
    -----------------------------------------------------------------------
     image and json keys        |               shot names                |
    -----------------------------------------------------------------------
     Asset name    | Asset type |  GT(shot0-4)     | test      | CG       |
    -----------------------------------------------------------------------
     image          image           yes, RGB         yes, RGB    yes, RGBA
     left_eye       bezier          yes, raw+revec   yes         no
     right_eye      bezier          yes, raw+revec   yes         no
     left_eyebrow   bezier          yes, raw+revec   yes         no
     right_eyebrow  bezier          yes, raw+revec   yes         no
     left_sideburn  bezier          yes, raw+revec   no          no
     right_sideburn bezier          yes, raw+revec   no          no
     left_seam      bezier          yes, raw         no          no
     nose_seam      bezier          yes, raw         no          no
     right_seam     bezier          yes, raw         no          no
     points         points          yes,     revec   yes         yes
     
     GT-revec (43 points, shot0-4) layout
                  13 14 15         21 22 23
               12          16   20          24
                  19 18 17         27 26 25       
       29 30        1   2          7  8        35 36  
     28   31  40  0       3  41 6        9 42  34    37    
       33 32        5   4          11 10       39 38
     
     CG (45 points, (0,0) implies occluded) layout
                  18 19 20         26 27 28
               17          21   25          29
          33      24 23 22         32 31 30     39   
       38 34        6   7          12 13        40 44 
          0   1   5       8  2  11       14  3  4       
       37 35        10  9          16 15        41 43 
          36                                    42   
          
     test (28 points) layout
                  7  8  9          23 22 21
                6          10   24          20
                  13 12 11         25 26 27   
                    1   2          16 15          
                  0       3  2  17       14             
                    5   4          18 19          
    '''

    @classmethod
    def asset_files(cls, shot, frame):
        dct = dict()
        dct['image_png'] = f'image-{shot}-frame{frame:03d}.png'
        dct['visual_png'] = f'visual-{shot}-frame{frame:03d}.png'
        if shot in ('shot0', 'shot1', 'shot2', 'shot3', 'shot4'):
            dct['raw_shapes_json'] = f'raw_shapes-{shot}-frame{frame:03d}.json'
            dct['shapes_json'] = f'revec_shapes-{shot}-frame{frame:03d}.json'
        if shot in ('test',):
            dct['shapes_json'] = f'shapes-{shot}-frame{frame:03d}.json'
        if shot in ('CG',):
            dct['shapes_json'] = f'points-{shot}-frame{frame:03d}.json'

        ret = dict()
        for file_type, file in dct.items():
            file = os.path.abspath(os.path.join(cls.target_dir, shot, file))
            if os.path.exists(file):
                ret[file_type] = file
            else:
                print(f'Warning: {file} not found')
                return dict()
        return ret

    @staticmethod
    def plot_points(orig_image,
                    points,
                    txt_color=(0, 255, 0), txt_width=1, txt_thick=1,
                    pts_color=(0, 255, 0), pts_wd=1,
                    alpha=0.8):
        image = orig_image.copy()
        for n, (x, y) in enumerate(points):
            x, y = int(x), int(y)
            if abs(x) + abs(y) > 0:
                cv2.putText(image, str(n), (x + 4, y - 4), cv2.FONT_HERSHEY_PLAIN,
                            txt_width, txt_color, txt_thick, cv2.LINE_AA)
            cv2.circle(image, (x, y), pts_wd, pts_color, -1)
        image = np.clip(np.round(alpha * image + (1.0 - alpha) * orig_image), 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def plot_bezier(orig_image,
                    bezier,
                    cnt_color=(0, 0, 255), cnt_wd=1,
                    alpha=0.8):
        image = orig_image.copy()
        cnt = []
        for segment in bezier:
            p0, t0, p3, t3 = np.array(segment).T
            p1, p2 = t0 + p0, p3 + t3
            stretch = np.linspace(0, 1, 20)
            for t in stretch:
                cnt.append((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)
        cnt = np.array(cnt).astype(np.int32)
        cv2.polylines(image, [cnt], False, cnt_color, cnt_wd)
        image = np.clip(np.round(alpha * image + (1.0 - alpha) * orig_image), 0, 255).astype(np.uint8)
        return image

    @classmethod
    def visualize(cls, shot='random', frame='random', plot=False):
        # Creates visual such as the ones in the data folder
        if shot=='random':
            shot = random.choice(list(cls.shot_frame_dct.keys()))
        if frame=='random':
            frame = random.randint(0, cls.shot_frame_dct[shot]-1)

        asset_files = cls.asset_files(shot, frame)

        print(f'Visualizing {shot}/{frame} with asset files {asset_files}')
        image_png = asset_files['image_png']
        visual = cv2.imread(image_png)

        shapes_json = asset_files['shapes_json']
        points = json.loads(open(shapes_json).read())['points']

        if shot in ('shot0', 'shot1', 'shot2', 'shot3', 'shot4'):
            raw_shapes = json.loads(open(asset_files['raw_shapes_json']).read())
            for name in cls.raw_bezier_names:
                visual = cls.plot_bezier(visual, raw_shapes[name], cnt_color=(255, 255, 0), cnt_wd=2, alpha=0.5)

            shapes = json.loads(open(asset_files['shapes_json']).read())
            for name in cls.revec_bezier_names:
                visual = cls.plot_bezier(visual, shapes[name], cnt_color=(0, 0, 255), cnt_wd=1, alpha=1)

        if shot in ('test'):
            shapes = json.loads(open(asset_files['shapes_json']).read())
            for name in cls.test_bezier_names:
                visual = cls.plot_bezier(visual, shapes[name], cnt_color=(0, 0, 255))

        visual = cls.plot_points(visual, points)
        cv2.putText(visual, f'{shot}/{frame:03}', (20, 780), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

        if plot:
            plt.figure(figsize=(15, 15))
            plt.imshow(visual[:, :, [2,1,0]])
            plt.axis('off')
            plt.show()
        return visual


    @classmethod
    def validate(cls):
        ok = True
        for shot, num_frames in cls.shot_frame_dct.items():
            if not ok:
                break
            for frame in range(num_frames):
                dct = cls.asset_files(shot, frame)
                if len(dct)==0:
                    ok = False
                    break
        print('Validation %s' % ('passed' if ok else 'failed'))
        return ok

    @classmethod
    def points_data(cls, mode='GT', batch_size=2):
        # supported modes -- GT (default), CG, test
        mode_shots_dct = {'GT': ('shot0', 'shot1', 'shot2', 'shot3'),
                          'CG': ('CG',),
                          'test': ('test',)}

        shots = mode_shots_dct[mode]
        image_files, points = [], []
        for shot in shots:
            for frame in range(cls.shot_frame_dct[shot]):
                dct = cls.asset_files(shot, frame)
                image_files.append(dct['image_png'])
                points.append(json.loads(open(dct['shapes_json']).read())['points'])

        zipped_lst = list(zip(image_files, points))
        random.shuffle(zipped_lst)
        image_files, points = zip(*zipped_lst)
        image_files, points = list(image_files), list(points)

        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(tf.io.read_file)
        channels = 4 if mode=='CG' else 3
        ds_image = ds.map(lambda x: tf.image.decode_png(x, channels=channels), num_parallel_calls=AUTOTUNE)
        ds_points = tf.data.Dataset.from_tensor_slices(points)

        ds = tf.data.Dataset.zip((ds_image, ds_points))
        ds = ds.shuffle(buffer_size=10)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    @classmethod
    def download_archive(cls):
        source_url = 'https://drive.google.com/drive/u/1/folders/15arH8QAIYQV_qcgUpU83MK3aBsW4Pnm9' # v1
        file = 'WACV_dataset_sir_lionel_1.zip'
        print(f'Manually copy {file} from {source_url} to specified folder {cls.target_dir}')


if __name__=='__main__':

    target_dir = '/Users/swarnend/Downloads/dataset/WACV_dataset_sir_lionel' # set your preferred dataset directory
    sir_lionel = SirLionel(target_dir=target_dir)

    if not sir_lionel.validate():
        sir_lionel.download_archive()
        sys.exit(0)

    write_file = 'sample.png'
    visual = sir_lionel.visualize(shot='random', frame='random')
    cv2.imwrite(write_file, visual)
    print(f'Written sample in {os.path.abspath(write_file)} for visual inspection')

    for mode in ('GT', 'test', 'CG'):
        dataset = SirLionel.points_data(mode=mode, batch_size=8)
        sample = 0
        for x, y in dataset:
            print(f'Mode {mode}, sample {sample+1}, image_tensor {x.numpy().shape}, points_tensor {y.numpy().shape}')
            sample += 1
            if sample==2:
                break
    print('tf.data loading successful!')
