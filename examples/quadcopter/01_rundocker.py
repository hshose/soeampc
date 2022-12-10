import subprocess
import time
import os
from datetime import datetime
from pathlib import Path

import fire

def rundocker(instances=16, samplesperinstance=int(1e5)):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    containertag = "soeampc:"+str(now)

    fp = Path(os.path.abspath(os.path.dirname(__file__)))
    soeampcdir = Path(os.path.abspath(fp.joinpath('..','..')))

    print("\n\n===============================================")
    print("Building Docker container", containertag)
    print("===============================================\n")

    os.chdir(soeampcdir)

    s = subprocess.Popen(["docker", "build",
            "-t", containertag,
            "."])
    if s.wait() != 0:
        print("Docker build error... exit")
        exit()

    print("\n\n===============================================")
    print("Running", instances, "Docker container to produce", float(samplesperinstance),"datapoints each")
    print("===============================================\n")

    os.chdir(fp)
    datasetpath = str(fp.joinpath(os.path.abspath(fp),'datasets'))
    print("datasetpath = ", datasetpath)
    processes = []
    for i in range(instances):
        command = " ".join(["python3", "01_samplempc.py", "--showplot=False", "--randomseed=None", "--experimentname=Docker_"+str(now)+"_"+str(i)+"_", "--numberofsamples="+str(samplesperinstance)])
        if i==instances-1:
            out = None
        else:
            out = subprocess.DEVNULL
        containername = "soeampc_samplempc_"+str(now)+"_"+str(i)
        p = subprocess.Popen(["docker", "run",
            "--rm",
            "--workdir", "/soeampc/examples/quadcopter",
            "-v", datasetpath+":/soeampc/examples/quadcopter/datasets",
            "--name", containername,
            containertag, "bash", "-c", command],
            stdout=out, stderr=out)
        processes.append(p)

    for p in processes:
        p.wait()

    Nsamples = int(instances*samplesperinstance)
    uid = os.getuid()
    gid = os.getgid()
    command = " ".join(["python3", "01_mergesamples.py", "mergedocker", "--now="+str(now), "--Nsamples="+str(Nsamples), "&&", "chown", "-R", str(uid)+":"+str(gid), "datasets" ])
    containername = "soeampc_mergesamples_"+str(now)
    p = subprocess.Popen(["docker", "run",
        "--rm",
        "--workdir", "/soeampc/examples/quadcopter",
        "-v", datasetpath+":/soeampc/examples/quadcopter/datasets",
        "--name", containername,
        containertag, "bash", "-c", command])
    p.wait()

if __name__=="__main__":
    fire.Fire(rundocker)