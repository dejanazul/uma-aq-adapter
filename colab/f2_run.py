import os
import subprocess

steps = ['bash /content/uma_zs_setup.sh'] if os.environ.get('F2_SETUP') == '1' else []
steps.append(
    f'cd /content && HF_TOKEN=$(cat /content/hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 '
    f'python /content/f2_eval.py {os.environ["F2_ARGS"]}'
)
script = ' && '.join(steps) + ' ; echo all_done'
log = open(os.environ.get('F2_LOG', '/content/f2.log'), 'w')
p = subprocess.Popen(['bash', '-c', script], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(f'f2 PID {p.pid}: args {os.environ["F2_ARGS"]}, setup {os.environ.get("F2_SETUP") == "1"}')
