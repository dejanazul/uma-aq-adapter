import os
import subprocess

steps = ['bash /content/film_setup.sh'] if os.environ.get('FILM_SETUP') == '1' else []
if os.environ.get('FILM_ARGS'):
    steps.append(
        f'cd /content && MPLBACKEND=Agg HF_TOKEN=$(cat /content/hf_token) HF_HUB_DISABLE_PROGRESS_BARS=1 '
        f'/content/film_venv/bin/python /content/film_eval.py {os.environ["FILM_ARGS"]}'
    )
script = ' && '.join(steps) + ' ; echo all_done'
log = open(os.environ.get('FILM_LOG', '/content/film.log'), 'w')
p = subprocess.Popen(['bash', '-c', script], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(f'film PID {p.pid}: setup {os.environ.get("FILM_SETUP") == "1"}, args {os.environ.get("FILM_ARGS")}')
