import random 
import subprocess
from os import listdir
from os.path import isfile, join
# 20 000 images

for i in range(0,20):
	to_call = [
		# "python",'single_video.py',
		"python",'single_video_pybullet2.py',
		'--spp','40000',
		'--nb_frames', '100',
		#'--nb_objects',str(int(random.uniform(50,75))),
		'--nb_objects',str(int(random.uniform(5,10))),
		# "--easy",
		#'--static_camera',
		# '--nb_frames', '1',
		# '--nb_objects',str(1),
		'--outf',f'dataset_redtea/{str(i).zfill(3)}',
        '--nb_distractors',str(2)
	]
	#print(f"dataset/{str(i).zfill(3)}")
	subprocess.call(to_call)
	#subprocess.call(['mv',f'dataset/{str(i).zfill(3)}/video.mp4',f"dataset/{str(i).zfill(3)}.mp4"])
	#break
	#cmd = (f"rsync -r output/dataset/{str(i).zfill(3)} output/output_example;rm -rf output/dataset/{str(i).zfill(3)}")
	#'rsync -r output/dataset/000 /mnt/adlr/dataset_2/ ;rm', '-rf', 'output/dataset/000'
	# subprocess.Popen(["rsync",'-r',f'output/dataset/{str(i).zfill(3)}',
	# 	"output/output_example",";",
	# 	'rm','-rf',f'output/dataset/{str(i).zfill(3)}'])
	# p= subprocess.Popen(["rsync",'-r',f'output/dataset/{str(i).zfill(3)}',
	# 	"output/output_example"])

	# onlyfiles = [f for f in listdir(f'output/dataset/{str(i).zfill(3)}') if isfile(join(f'output/dataset/{str(i).zfill(3)}', f))]
	# p= subprocess.Popen(["mv",'-r',f'output/dataset/{str(i).zfill(3)}',
	# 	"output/output_example"])
	# p.wait()
	# subprocess.Popen(['rm','-rf',f'output/dataset/{str(i).zfill(3)}'])
	#subprocess.Popen(cmd.split())
	#print(cmd)
	
	#break
