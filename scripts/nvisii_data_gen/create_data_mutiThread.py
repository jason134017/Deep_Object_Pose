
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
    default= 0,
    help='start dir index')

parser.add_argument("--end",
    default= 1,
    help='start dir index')
parser.add_argument("--thread",
    default= 1,
    help='start dir index')
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
opt = parser.parse_args()
print(str(opt.outf))

commands = []
# 20 000 images
for i in range(opt.start,opt.end):
    
    # commands.append(f'python single_video_pybullet2.py --spp 10000 --nb_frames 100 --nb_objects {str(int(random.uniform(5,10)))} --outf dataset_redtea/{str(i).zfill(3)} --nb_distractors 0')
    commands.append(f'python single_video_pybullet.py --spp 10000 --nb_frames 100 --outf {str(opt.outf)}/{str(i).zfill(3)}  \
--objs_folder {opt.objs_folder} --nb_objects {opt.nb_objects} --nb_distractors {opt.nb_distractors} \
--objs_folder_distrators {opt.objs_folder_distrators} --skyboxes_folder {opt.skyboxes_folder}')
    # print(commands[-1])
    #com = f'echo {str(i)}'
    #commands.append(com)
pool = Pool(opt.thread) # two concurrent commands at a time
for i, returncode in enumerate(pool.imap(partial(call, shell=True), commands)):
    if returncode != 0:
       print("%d command failed: %d" % (i, returncode))
