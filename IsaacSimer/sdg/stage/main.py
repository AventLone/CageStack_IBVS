from isaacsim import SimulationApp
import os
import sys
import asyncio
from actor import ActorSDG

BASE_EXP_PATH = os.path.join(os.environ["EXP_PATH"], "isaacsim.exp.action_and_event_data_generation.base.kit")
APP_CONFIG = {"renderer": "RayTracedLighting", "headless": False, "width": 1920, "height": 1080}

def actorHuman(sim_app):
    
    ### 默认命令文档
    config_file_path = "./default_config.yaml"
    no_random_commands = True
    
    print("Config file path: {}".format(config_file_path))
    print("Don't random commands: {}".format(no_random_commands))

    # Check files exist
    if not os.path.isfile(config_file_path):
        print("Invalid config file path. Exit.", file=sys.stderr)
        return

    # Start SDG
    sdg = ActorSDG(
        sim_app,
        os.path.abspath(config_file_path),
        no_random_commands,
    )

    from omni.kit.async_engine import run_coroutine

    task = run_coroutine(sdg.run())
    # try:
    while not task.done():
        sim_app.update()

    if not task.result():
        print("Failed to run SDG")

    # # Close app
    # finally:
    #     sim_app.update()



# def main_func():
#     simulation_app = SimulationApp(launch_config=APP_CONFIG, experience=BASE_EXP_PATH)

#     import random, numpy as np, omni.replicator.core as rep
#     from omni.kit.async_engine import run_coroutine

#     # 设置随机种子
#     seed = int.from_bytes(os.urandom(8), "little") % (2**32)
#     random.seed(seed)
#     np.random.seed(seed)
#     rep.set_global_seed(seed)
#     print(f"[INFO] Using random seed: {seed}")

#     from Env import enviroment, ResolveJson
    
#     configPath = "./config.json"
#     config = ResolveJson(configPath)
    
#     pathlist = []
#     science = 1

#     Env = enviroment(pathlist, config, science)
#     Env.CreateWarehouse()
#     Env.RandomUpdateMap()

#     # 启动人物动画 ActorSDG（异步任务）
#     config_file_path = "./default_config.yaml"
#     no_random_commands = True
#     sdg = ActorSDG(simulation_app, os.path.abspath(config_file_path), no_random_commands)
#     task_actor = run_coroutine(sdg.run())  # 异步执行，不阻塞

#     # 启动数据保存（同步逻辑）
#     write_task = None
#     if science:
#         write_task = run_coroutine(async_write_images(Env))
#     else:
#         Env.WriteImages()

#     try:
#         # 模拟循环：保持 IsaacSim 运行直到任务结束
#         while True:
#             simulation_app.update()
#             actor_done = task_actor.done()
#             write_done = write_task.done()
#             if write_done:
#                 break
            

#         print("✅ Both Actor animation and image writing finished.")
#     finally:
#         simulation_app.close()


# async def async_write_images(Env):
#     await asyncio.sleep(2.0)  # 等场景加载稳定
#     Env.WriteMoveImages()

def main_func():

    import random, numpy as np, omni.replicator.core as rep


    from Env import enviroment, ResolveJson
    
    ### 配置文件
    configPath = "./config.json"
    config = ResolveJson(configPath)
    
    ### 路径 List[Tuple(float, float)]
    pathlist = []
    
    ### 场景 0：明眸， 1：货架
    science=1
    
    ### 场景初始化&仓库创建&物体随机布置
    Env = enviroment(pathlist, config, science)
    Env.CreateWarehouse()
    Env.RandomUpdateMap()
    
    ### 人物驱动
    # actorHuman(simulation_app)
    
    ### 数据保存
    if science:
        Env.WriteMoveImages()
    else:
        Env.WriteImages()
    
    Env.Close_stage()



if __name__ == "__main__":
    # 启动 Isaac Sim 应用
    simulation_app = SimulationApp(launch_config=APP_CONFIG, experience=BASE_EXP_PATH)
    # count = 0
    # while count < 1:
    #     count += 1
    #     main_func()
    main_func()
    
    while simulation_app.is_running():
        simulation_app.update()
        # simulation_app.close()