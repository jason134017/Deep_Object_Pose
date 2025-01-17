
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import random 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--samedir",
    action='store_true',
    help='store in same dir')
parser.add_argument("--start",
    type=int,
    default= 0,
    help='start dir index')

parser.add_argument("--end",
    type=int,
    default= 1,
    help='start dir index')
parser.add_argument("--thread",
    default= 1,
    help='start dir index')
parser.add_argument(
    '--nb_frames',
    default=2000,
    help = "how many frames to save"
)
parser.add_argument(
    '--nb_objects',
    default=28,
    type = int,
    help = "how many objects"
)
parser.add_argument(
    '--nb_distractors',
    default=15,
    help = "how many objects"
)
parser.add_argument(
    '--spp',
    default=4000,
    type=int,
    help = "number of sample per pixel, higher the more costly"
)
parser.add_argument(
    '--outf',
    default='output_test',
    help = "output filename inside output/"
)

parser.add_argument(
    '--objs_folder_distrators',
    default='google_scanned_models/',
    help = "object to load folder"
)
parser.add_argument(
    '--objs_folder',
    default='models/',
    help = "object to load folder"
)
parser.add_argument(
    '--path_single_obj',
    default=None,
    help='If you have a single obj file, path to the \
    obj directly.'
)
parser.add_argument(
    '--scale_single_obj',
    default=1,
    type=float,
    help='change the scale of the path_single_obj loaded.'
)

parser.add_argument(
    '--skyboxes_folder',
    default='dome_hdri_haven/',
    help = "dome light hdr"
)

parser.add_argument(
    '--width',
    default=500,
    type=int,
    help = 'image output width'
)
parser.add_argument(
    '--height',
    default=500,
    type=int,
    help = 'image output height'
)

opt = parser.parse_args()


commands = []
# 20 000 images
for i in range(opt.start,opt.end):
    
    # commands.append(f'python single_video_pybullet2.py --spp 10000 --nb_frames 100 --nb_objects {str(int(random.uniform(5,10)))} --outf dataset_redtea/{str(i).zfill(3)} --nb_distractors 0')
    commands.append(f'python single_video_pybullet.py --spp {opt.spp} --nb_frames {opt.nb_frames} --outf {str(opt.outf)}/{str(i).zfill(3)}  \
--objs_folder {opt.objs_folder} --nb_objects {opt.nb_objects} --nb_distractors {opt.nb_distractors} \
--objs_folder_distrators {opt.objs_folder_distrators} --skyboxes_folder {opt.skyboxes_folder} --height {opt.height} --width {opt.width}')
    # print(commands[-1])
    #com = f'echo {str(i)}'
    #commands.append(com)
pool = Pool(opt.thread) # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))
