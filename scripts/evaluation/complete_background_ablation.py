"""Continue the explicitly authorized full experiment; fail closed on any stage error.

Waits for separately launched validation/control preparation, then runs the
remaining dependent stages. Every child PID, command and log is recorded so a
long experiment is inspectable. Does not delete or overwrite historical data.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.experiment.resolve()
    if not experiment.is_relative_to(ROOT/'data/processed'):
        raise ValueError('Unexpected experiment path')
    logs = experiment/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    if (experiment/'orchestrator.json').exists():
        raise FileExistsError('An orchestration record already exists; inspect it before starting another')
    state = dict(complete=False, phase='waiting_validation', started_unix=time.time(), pid=os.getpid(), children=[])
    handles = []
    children = []
    def update(phase):
        state.update(phase=phase,updated_unix=time.time())
        (experiment/'orchestrator.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        print(phase,flush=True)
    def spawn(label, stage, *extra, script='run_background_ablation.py'):
        command = [sys.executable,'-u',str(ROOT/'scripts/evaluation'/script)]
        if stage:
            command.append(stage)
        command += ['--experiment',str(experiment),*extra]
        log = (logs/(label+'.log')).open('x',encoding='utf-8')
        handles.append(log)
        process = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append(process)
        state['children'].append(dict(label=label,pid=process.pid,command=command,log=str(log.name)))
        update('started '+label)
        return process
    def wait(process, label):
        while process.poll() is None:
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f'{label} failed with exit {process.returncode}; inspect {logs/(label+".log")}')
        update('completed '+label)
    def marker(path):
        return path.exists()
    try:
        update('waiting for frozen validation and control preparation')
        deadline = time.monotonic()+7200
        required = [experiment/'validation'/s/'report.json' for s in ('90000002','90000003')]
        required += [experiment/'models/old64_single/manifests/paper_config.json']
        while not all(marker(p) for p in required):
            if time.monotonic()>deadline:
                raise TimeoutError('Validation/preparation did not complete; no training started')
            time.sleep(10)
        wait(spawn('select','select'),'select')
        selected = json.loads((experiment/'selection.json').read_text(encoding='utf-8'))['selected']
        if selected == 'old64_single':
            raise RuntimeError('No new candidate passed the frozen background gate; training not started')
        build = spawn('build_new','build')
        control = spawn('train_control','train','--method','old64_single')
        wait(build,'build_new')
        wait(spawn('prepare_new','prepare','--method',selected),'prepare_new')
        wait(control,'train_control')
        wait(spawn('train_new','train','--method',selected),'train_new')
        wait(spawn('infer','infer'),'infer')
        wait(spawn('calibrate','calibrate'),'calibrate')
        sequences = ('90000002','90000003','90000004','90000005_1','90000005_2')
        # Three concurrent GPU inference workers fit comfortably below the
        # single training worker's VRAM; limit CPU/RAM rather than run all five.
        active = []
        for sequence in sequences:
            if len(active)>=3:
                process,label = active.pop(0)
                wait(process,label)
            label = 'test_'+sequence
            active.append((spawn(label,'test','--sequence',sequence),label))
        for process,label in active:
            wait(process,label)
        wait(spawn('report',None,script='report_background_ablation.py'),'report')
        state['complete'] = True
        update('all stages complete')
    except BaseException as error:
        state['error'] = repr(error)
        # Do not leave sibling GPU/background jobs running after a failed gate.
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        update('failed; see error and child logs')
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
