import subprocess
import time
import os
from datetime import datetime

import fire

def rundocker(instances=16, samplesperinstance=int(1e5)):
    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    instances = 16
    samplesperinstance = int(1e5)
    containertag = "soeampc:batch"

    os.chdir("/home/hose/projects/dsme/soeampc")

    s = subprocess.Popen(["docker", "build",
            "-t", containertag,
            "."])
    if s.wait() != 0:
        print("Docker build error... exit")
        exit()

    os.chdir("/home/hose/projects/dsme/soeampc/examples/quadcopter")
    processes = []
    for i in range(instances):
        command = " ".join(["python3", "01_samplempc.py", "--showplot=False", "--randomseed=None", "--experimentname=Docker_"+str(now)+"_"+str(i)+"_", "--numberofsamples="+str(samplesperinstance)])
        if i==instances-1:
            out = None
        else:
            out = subprocess.DEVNULL
        containername = "soeampc_"+str(i)
        p = subprocess.Popen(["docker", "run",
            "--rm",
            "--workdir", "/soeampc/examples/quadcopter",
            "-v", "/home/hose/projects/dsme/soeampc/examples/quadcopter/datasets:/soeampc/examples/quadcopter/datasets",
            "--name", containername,
            containertag, "bash", "-c", command],
            stdout=out, stderr=out)
        processes.append(p)

    for p in processes:
        p.wait()

    Nsamples = int(instances*samplesperinstance)
    command = " ".join(["python3", "01_mergesamples.py", "--now="+str(now), "--Nsamples="+str(Nsamples)])
    containername = "soeampc_merge"
    p = subprocess.Popen(["docker", "run",
        "--rm",
        "--workdir", "/soeampc/examples/quadcopter",
        "-v", "/home/hose/projects/dsme/soeampc/examples/quadcopter/datasets:/soeampc/examples/quadcopter/datasets",
        "--name", containername,
        containertag, "bash", "-c", command])

    p.wait()

if __name__=="__main__":
    fire.Fire(rundocker)