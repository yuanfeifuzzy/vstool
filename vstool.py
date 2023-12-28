#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Useful functions or Classes as utilities for using in virtual screening
"""

import os
import sys
import time
import traceback
from pathlib import Path
import importlib.metadata
from functools import partial
from datetime import timedelta
from multiprocessing import Queue, Pool, cpu_count

import cmder
import pandas as pd
from loguru import logger


def setup_logger(quiet=False, verbose=False, formatter='<level>[{time:YYYY-MM-DD HH:mm:ss}] {message}</level>'):
    logger.remove()
    level = "DEBUG" if verbose else ("ERROR" if quiet else "INFO")
    logger.add(sys.stdout, colorize=True, format=formatter, level=level)
    return logger


setup_logger()


def get_version(package):
    v = '0.0.0'
    if package:
        try:
            v = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as e:
            logger.error(e)
    else:
        logger.debug(f'Package is None, default version {v} was returned')
    return v


def error_and_exit(message, exit_code=1, task=0, status=0, **kwargs):
    logger.error(message)
    if task:
        error = kwargs.pop('error', message)
        task_update(task, status, error=error, **kwargs)
    sys.exit(exit_code)
    

def info_and_exit(message, exit_code=0, task=0, status=0, **kwargs):
    logger.info(message)
    if task:
        task_update(task, status, **kwargs)
    sys.exit(exit_code)


def debug_and_exit(message, exit_code=0, task=0, status=0, **kwargs):
    logger.debug(message)
    if task:
        task_update(task, status, **kwargs)
    sys.exit(exit_code)


def task_update(task, status, **kwargs):
    exe = 'task_update'
    p = cmder.run(f'which {exe}', log_cmd=False)
    if p.returncode:
        logger.warning('No executable for task update was found, update status ignored')
    else:
        if task:
            cmd = f'{exe} {task} {status}'
            for k, v in kwargs.items():
                if k == 'error':
                    cmd = f"{cmd} --{k} '{v}'"
                else:
                    cmd = f'{cmd} --{k} {v}'
            try:
                cmder.run(cmd, fmt_cmd=False)
            except Exception as e:
                error_and_exit(f'Failed to update task {task} with status {status} due to {e}', 1)


def get_available_gpus(gpus=0, exit_code=1, task=0, status=-4):
    ids, computers = set(), set()
    p = cmder.run(f'nvidia-smi', log_cmd=False)
    if p.returncode:
        error_and_exit(f'Failed to get GPU status', exit_code, task=task, status=status)
    else:
        lines, i = p.stdout.read().splitlines(), 0
        for i, line in enumerate(lines):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                ids.add(fields[1])
            if 'GPU   GI   CI        PID   Type   Process name' in line:
                break
        for line in lines[i + 3:-1]:
            fields = line.split()
            idx, mode = fields[1], fields[5]
            if mode == 'C':
                computers.add(idx)

    ids = list(ids - computers)
    return ids[:gpus] if gpus and len(ids) >= gpus else ids


def get_available_cpus(cpus=0):
    cpus = min(cpus or cpu_count(), cpu_count())
    cpus = os.environ.get('SLURM_CPUS_PER_TASK', cpus)
    return int(cpus)


def parallel_cpu_task(func, iterable, processes=None, chunksize=1, **kwargs):
    """
    Process iterable using function in parallel mode on multiple CPU (processes).

    :param func: function
    :param iterable: iterable
    :param processes: int, maximum number of processes
    :param chunksize: int, or very long iterables using a large value for chunksize can make the job complete
        much faster than using the default value of 1
    :param kwargs: dict, other arguments need to be passed to func
    :return: list
    """

    with Pool(processes=get_available_cpus(cpus=processes)) as pool:
        func = partial(func, **kwargs) if kwargs else func
        return list(pool.imap(func, iterable, chunksize=chunksize))


def gpu_queue(n=None, exit_code=1, task=0, status=0):
    """
    A Queue object stores indices of available GPUs

    :return: Queue
    """

    gpus = get_available_gpus(exit_code=exit_code, task=task, status=status)
    gpus = gpus[:n] if n and n <= len(gpus) else gpus
    queue = Queue()
    for gpu in gpus:
        queue.put(gpu)
    return queue


def parallel_gpu_task(func, iterable, **kwargs):
    """
    Process iterable using function in parallel mode on multiple GPUs.

    :param func: function
    :param iterable: iterable
    :param kwargs:
    :return: list
    """

    queue = kwargs.get('gpu_queue', Queue())
    processes = queue.qsize() or len(get_available_gpus())
    if processes >= 1:
        logger.debug(f'Processing tasks with {processes} GPUs')
        results = parallel_cpu_task(func, iterable, processes, **kwargs)
        if kwargs.get('check_return_code', False):
            n, m = len(results), sum(bool(result) for result in results)
            if m > 0:
                error_and_exit(f'Submitted {n:,} tasks to {processes} GPUs, {m:,} tasks return non-zero code, '
                               f'check the log for details', 1)
    else:
        error_and_exit('No GPU or visible GPU was found', 1)
        results = []
    return results


def check_exist(path, message='', exit_code=1, task=0, status=-1):
    if path:
        try:
            if not Path(path).exists():
                error_and_exit(message or f'Path {path} does not exist', exit_code, task=task, status=status)
        except Exception as e:
            error_and_exit(f'Check path {path} failed due to {e}', exit_code, task=task, status=status)
    else:
        error_and_exit(f'Path is invalid, cannot continue', exit_code, task=task, status=status)
    return Path(path).resolve()


def check_file(path, message='', exit_code=1, task=0, status=-1):
    if path:
        try:
            if not Path(path).is_file():
                error_and_exit(message or f'File {path} is not a file or does not exist', exit_code, task=task,
                               status=status)
        except Exception as e:
            error_and_exit(f'Check file {path} failed due to {e}', exit_code, task=task, status=status)
    else:
        error_and_exit(f'File is invalid, cannot continue', exit_code, task=task, status=status)
    return Path(path).resolve()


def check_dir(path, message='', exit_code=1, task=0, status=-1):
    if path:
        try:
            if not Path(path).is_dir():
                error_and_exit(message or f'Directory {path} is not a directory or does not exist', exit_code,
                               task=task, status=status)
        except Exception as e:
            error_and_exit(f'Check directory {path} failed due to {e}', exit_code, task=task, status=status)
    else:
        error_and_exit(f'Directory is invalid, cannot continue', exit_code, task=task, status=status)
    return Path(path).resolve()


def check_exe(exe, message='', exit_code=1, task=0, status=-2):
    if exe:
        p = cmder.run(f'which {exe}', log_cmd=False)
        if p.returncode:
            error_and_exit(message or f'Executable {exe} does not exist', exit_code, task=task, status=status)
        else:
            return p.stdout.read().strip()
    else:
        error_and_exit(f'Executable is None, cannot continue', exit_code, task=task, status=status)


def mkdir(directory, exit_code=1, task=0, status=-1):
    d = Path(directory or '.').resolve()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except IOError as e:
        error_and_exit(f'Failed to make directory {d} due to {e}', exit_code, task=task, status=status)
    return d


def get_extension(filename, extensions=None):
    extensions = extensions or ('.tsv', '.tsv.gz', '.csv', '.csv.gz', '.parquet', '.smi', '.smiles')
    for extension in extensions:
        if str(filename).lower().endswith(extension):
            break
    else:
        extension = ''
    return extension


def read(filename, exit_code=1, task=0, status=0, **kwargs):
    """
    Read a file into pd.DataFrame (file type was determined using file extension)
    """

    extension = get_extension(filename)
    if not extension:
        error_and_exit(f'Failed to read {filename} due to unable to guess file type based on extension',
                       exit_code, task=task, status=status)
    readers = {
        '.csv': pd.read_csv,
        '.csv.gz': pd.read_csv,
        '.tsv': partial(pd.read_csv, sep='\t'),
        '.tsv.gz': partial(pd.read_csv, sep='\t'), '.parquet': pd.read_parquet,
        '.smi': partial(pd.read_csv, sep='\t', haeder=None, names=['SMILE', 'UID']),
        '.smiles': partial(pd.read_csv, sep='\t', haeder=None, names=['SMILE', 'UID'])
    }
    reader = readers[extension]
    return reader(filename, **kwargs)


def df_to_smiles(output, df=pd.DataFrame(), smiles_column='SMILES', name_column='index'):
    if df.empty:
        logger.warning(f'Empty dataframe found, no SMILES will be saved!')
    else:
        df = df[df[smiles_column] != '']
        if df.empty:
            logger.warning(f'All SMILES are empty, none of them will be saved!')
        else:
            if name_column == 'index':
                df.reset_index(inplace=True)
            df.to_csv(output, sep='\t', index=False, header=False, columns=[smiles_column, name_column])


def write(df, output, exit_code=1, task=0, status=0, **kwargs):
    """
    Write a pd.DataFrame into a file (file type was determined using file extension)
    """

    extension = get_extension(output)
    error_and_exit(f'Failed to write to {output} due to unable to guess file type based on extension',
                   exit_code, task=task, status=status)
    writers = {'.csv': partial(df.to_csv, index=False), '.csv.gz': partial(df.to_csv, index=False),
               '.tsv': partial(df.to_csv, sep='\t', index=False), '.tsv.gz': partial(df.to_csv, sep='\t', index=False),
               '.parquet': df.to_parquet, '.smi': partial(df_to_smiles, df=df), '.smiles': partial(df_to_smiles, df=df),
               }
    writer = writers[extension]
    writer(output, **kwargs)


def submit(cmd, nodes=1, ntasks=1, ntasks_per_node=1, job_name='vs',
           day=0, hour=8, minute=0, partition='', dependency='', mode='',
           email='', email_type='ALL', mail='', mail_type='ALL', log='%x.log',
           script='submit.sh', hold=False, delay=0, project=''):
    try:
        script = Path(script).resolve()
        logger.debug(f'Generating submission script {script}')
        with script.open('w') as o:
            o.write('#!/usr/bin/env bash\n')
            o.write('\n')
            o.write(f'#SBATCH --nodes={nodes}\n')
            o.write(f'#SBATCH --ntasks={ntasks}\n')
            o.write(f'#SBATCH --ntasks-per-node={ntasks_per_node}\n')
            o.write(f'#SBATCH --output={log}\n')
            if mode:
                o.write(f'#SBATCH --open-mode={mode}\n')

            o.write(f'#SBATCH --job-name={job_name}\n')
            o.write(f'#SBATCH --time={day}-{hour}:{minute}\n')

            if partition:
                o.write(f'#SBATCH --partition={partition}\n')

            if project:
                o.write(f'#SBATCH --account {project}\n')

            if dependency:
                o.write(f'#SBATCH --dependency={dependency}\n')
                o.write(f'#SBATCH --kill-on-invalid-dep=yes\n')

            if email or mail:
                o.write(f'#SBATCH --mail-user={email or mail}\n')
                o.write(f'#SBATCH --mail-type={email_type or mail_type}\n')

            if delay:
                o.write(f'#SBATCH --begin=now+{delay}{"hours" if delay > 1 else "hour"}\n')

            o.write(f'\n{cmd}\n\n')

        logger.debug(f'Successfully generated submit script {script}')

        if hold:
            logger.debug(f'Script {script.name} has not been submitted due to hold is True\n')
            return 0, 0
        else:
            p = cmder.run(f'sbatch {script.name}', cwd=script.parent)
            if p.returncode:
                logger.debug(f'Failed to submit {script.name} to job queue due to an error\n')
                return p.returncode, 0
            else:
                s = p.stdout.read()
                try:
                    sid = int(s.strip().splitlines()[-1].split()[-1])
                    logger.debug(f'Successfully submitted job with slurm job id {sid}\n')
                    return 0, sid
                except Exception as e:
                    logger.error(f'Failed to get job id from submit result due to {e}:\n\n{s}')
                    return 0, 0
    except Exception as e:
        logger.error(f'Failed to generate submit script and submit the job due to\n{e}\n\n{traceback.format_exc()}\n')
        return 1, 0


def profile(task=0, status=0, error_status=0, task_name='Task'):
    def inner(func):
        def wrapper(*args, **kwargs):
            try:
                start = time.time()

                x = func(*args, **kwargs)

                t = str(timedelta(seconds=time.time() - start))
                logger.debug(f'{task_name} complete in {t.split(".")[0]}\n')
                if task:
                    task_update(task, status)
                return x
            except Exception as e:
                error_and_exit(f'{task_name} failed due to\n{e}\n\n{traceback.format_exc()}\n',
                                       task=task, status=error_status or -status)

        return wrapper

    return inner


if __name__ == '__main__':
    pass
