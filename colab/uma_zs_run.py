import os
import subprocess

limit = os.environ.get('UMA_LIMIT', '1000000')
steps = ['bash /content/uma_zs_setup.sh'] if os.environ.get('UMA_SETUP') == '1' else []
steps.append(
    f'cd /content && HF_TOKEN=$(cat /content/hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 '
    f'python /content/uma_zs_eval.py --limit {limit}'
)
script = ' && '.join(steps) + ' ; echo all_done'
log = open('/content/uma_zs.log', 'w')
p = subprocess.Popen(['bash', '-c', script], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(f'stage Z PID {p.pid}: limit {limit}, setup {os.environ.get("UMA_SETUP") == "1"}')
