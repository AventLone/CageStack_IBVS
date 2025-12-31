from utils.common import *
simulate_app = open_simu_app("./configs/sdg_1.yaml")

from sdg.obj import ObjSDG

data_collect_config = load_config("./configs/sdg_1.yaml")["obj_data_collect"]

generator = ObjSDG(data_collect_config)
generator.generate()

while simulate_app.is_running() and not generator.is_finished:
    simulate_app.update()

simulate_app.close()