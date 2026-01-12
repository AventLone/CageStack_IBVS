from utils.common import *
simulate_app = open_simu_app("./configs/sdg_1.yaml")

from sdg.obj import ObjSDG

generator = ObjSDG(obj_prim_path="", frames_required=10)
generator.generate()

while simulate_app.is_running() and not generator.is_finished:
    simulate_app.update()

simulate_app.close()