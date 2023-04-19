import jinja2
import numpy as np
import fire

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
from pathlib import Path

fp = Path(os.path.dirname(__file__))
os.chdir(fp)

def renderdynamics(n_mass):
    xref = np.genfromtxt(fp.joinpath('mpc_parameters','xref_'+str(n_mass)+'.txt'), delimiter=',').flatten()
    environment = jinja2.Environment(loader=jinja2.FileSystemLoader(fp.joinpath("dynamics")))
    template = environment.get_template("f.template.py")
    filename = "f_"+str(n_mass)+".py"
    content = template.render(
        xref=str(xref.tolist())
    )
    with open(fp.joinpath("dynamics", filename), mode="w", encoding="utf-8") as file:
        file.write(content)
        # print(f"... wrote {filename}")


if __name__=="__main__":
    fire.Fire(renderdynamics)