import os
import subprocess

log = open(os.environ['LOG'], 'w')
p = subprocess.Popen(
    ['bash', '-c', os.environ['JOB'] + ' ; echo all_done'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
)
print(f'started PID {p.pid}: {os.environ["JOB"]} -> {os.environ["LOG"]}')
