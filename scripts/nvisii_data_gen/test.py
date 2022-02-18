import subprocess


for i in range(1):
    # subprocess.Popen(["rsync",'-r',f'output/dataset/{str(i).zfill(3)}',
    #     "output/output_example",";",
    #     'rm','-rf',f'output/dataset/{str(i).zfill(3)}'])
    cmd = ['rsync','-r','output/dataset/000}',
        'output/output_example']
    print(cmd)
    # subprocess.Popen(['rsync','-r','output/dataset/000}',
    #     'output/output_example',';',
    #     'rm','-rf','output/dataset/000'])
    subprocess.Popen(['rsync','-r','output/dataset/000',
        'output/output_example'])