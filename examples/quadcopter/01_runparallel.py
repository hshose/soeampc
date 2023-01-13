import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

import fire

def run_parallel(instances=16, samplesperinstance=int(1e5), prefix="Cluster"):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    containertag = "soeampc:"+str(now)

    fp = Path(os.path.abspath(os.path.dirname(__file__)))
    print("\n\n===============================================")
    print("Running", instances, "as process to produce", float(samplesperinstance),"datapoints each")
    print("===============================================\n")

    os.chdir(fp)
    datasetpath = str(fp.joinpath(os.path.abspath(fp),'datasets'))
    print("datasetpath = ", datasetpath)
    processes = []
    for i in range(instances):
        # command = ["python3", "01_samplempc.py", "--showplot=False", "--randomseed=None", "--experimentname=Docker_"+str(now)+"_"+str(i)+"_", "--numberofsamples="+str(samplesperinstance)]
        experimentname = prefix+"_"+str(now)+"_"+str(i)+"_"
        command = [
            "python3",
            "01_samplempc.py",
            "--showplot=False",
            "--randomseed=None",
            "--experimentname="+experimentname,
            "--numberofsamples="+str(samplesperinstance),
            "--generate=False"]

        with open(fp.joinpath('logs',experimentname+".log"),"wb") as out:
            p = subprocess.Popen(command,
                stdout=out, stderr=out)
            processes.append(p)

    for p in processes:
        p.wait()

if __name__=="__main__":
    fire.Fire(run_parallel)