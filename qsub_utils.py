import os
from random import randint
import time
from datetime import datetime


# Nifti Image I/O
def create_dir(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)   

# run code parallelly on local machine
def run_local_para(cmdStr_list, para_num=1, waite_time=60):

    '''
    cmdStr_list: collection of cmd to run on local machine (Each program should have roughly same running time)
    para_num: number of cmd to run parallelly
    waite_time: waiting time for parallel jobs to finish
    '''
    print(f'{len(cmdStr_list)} jobs to run, para_num = {para_num}.')

    time.sleep(5)
    for cidx, cmdStr in enumerate(cmdStr_list):
        if (cidx+1) % para_num == 0:
            print(f'{cmdStr}') # run cmdStr in foreground
            os.system(f'{cmdStr}') # run cmdStr in foreground
            time.sleep(waite_time)
        else:
            print(f'{cmdStr} &') # run cmdStr in background
            os.system(f'{cmdStr} &') # run cmdStr in background

    print('All jobs are done!')


# run code via qsub

# runnew.q

def run7(job_name, cmd_str, h_vmem=8, node='compute7.q', tmp_path=None, write_cmd_log=True, waite_time=0, job_num_lim=1500, verbosome=True):
    '''qsub job to the CenOS7

    node='iniadmin7.q' or 'compute7.q' or 'runnow.q'

    '''
    if tmp_path is None:
        tmp_path = '/ifs/loni/faculty/shi/spectrum/yxia/tmp/qsubs'


    create_dir(tmp_path)

    job_id_num = randint(0,10000000000)

    output_log =  os.path.join(tmp_path, f'{job_name}_{job_id_num}.o')
    error_log =  os.path.join(tmp_path, f'{job_name}_{job_id_num}.e')

    # write the data cmd for rerun
    if write_cmd_log:
        cmd_log =  os.path.join(tmp_path, f'{job_name}_{job_id_num}.cmd')
        os.system(f'echo "{cmd_str}" > {cmd_log}') 

    if (h_vmem != 8) and (node != 'iniadmin7.q'):
        node_info = f'-l h_vmem={h_vmem}G -q {node}'
    else:
        node_info = f'-q {node}'

    qsubStr = 'echo \'' + cmd_str + f'\' | qsub {node_info} -N ' + job_name + ' -o ' + output_log + ' -e ' +  error_log
    
    if verbosome:
        print(qsubStr)

    if waite_time > 30:

        # Get the current time
        current_time = datetime.now()
        print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
        qsub_waiter(username='yxia', job_num_lim=job_num_lim, waite_time=waite_time)

    os.system(qsubStr)


# check 
def write_job_status(status_file):
    cmdStr = 'qstat -u yxia>' + status_file
    os.system(cmdStr)


def clean_jobs():
    status_file = '/ifs/loni/faculty/shi/spectrum/yxia/tools/Code/python_module/job_status.txt'
    write_job_status(status_file)
    job_list=list()
    with open(status_file,'r') as fin:
        for line in fin:
            job_list.append(line)
    job_to_clean = set()
    for job in job_list:
        job_item = job.split()
        if len(job_item)>4:
            job_status = job.split()[4]
            if job_status == 'Eqw':
                job_to_clean.add(job.split()[0])
    job_id_list = ''
    for job_id in job_to_clean:
        job_id_list = job_id_list + job_id + ','

    cmdStr = 'qdel ' + job_id_list
    os.system(cmdStr)


import subprocess
import pandas as pd

def parse_qstat_output(output, username=None):
    lines = output.strip().split('\n')
    headers = lines[0].split()
    jobs_data = []

    for line in lines[2:]:  # Skip the header and the second line which is usually a separator
        values = line.split()
        job_info = dict(zip(headers, values))
        jobs_data.append(job_info)
        
    jobs_df = pd.DataFrame(jobs_data)
    
    if username is not None:
        jobs_df = jobs_df[jobs_df['user'] == username]

    return jobs_df

def read_qstat(username=None):
    result = subprocess.run(['qstat'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.stderr:
        print("Error running qstat:", result.stderr)
        return None

    return parse_qstat_output(result.stdout, username)


def qsub_waiter(username='yxia', job_num_lim=2500, waite_time=600):

    jobs_df = read_qstat(username)
    job_num = len(jobs_df)

    while job_num > job_num_lim:
        print(f'There are {job_num} running, waite for {waite_time}s !')
        time.sleep(waite_time)
        jobs_df = read_qstat(username)
        job_num = len(jobs_df)