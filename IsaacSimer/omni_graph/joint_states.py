import omni.graph.core as og

GRAPH_PATH = "/Joint_States"
ROBOT_PRIM_PATH = "/World/lola"
JOINT_STATE_TOPIC_NAME = "lola/joint_states"
JOINT_COMMAND_TOPIC_NAME = "lola/joint_command"

keys = og.Controller.Keys
(joint_states_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": GRAPH_PATH,
        "evaluator_name": "execution"
    },
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            ("SimulationTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

            ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
            ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
            ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

            ("SimulationTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ("SimulationTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

            ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
            ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
            ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
            ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand")
        ],
        keys.SET_VALUES: [            
            ("PublishJointState.inputs:targetPrim", ROBOT_PRIM_PATH),
            ("PublishJointState.inputs:topicName", JOINT_STATE_TOPIC_NAME),

            ("SubscribeJointState.inputs:topicName", JOINT_COMMAND_TOPIC_NAME),
            ("ArticulationController.inputs:targetPrim", ROBOT_PRIM_PATH)
        ]
    }
)

# import omni.graph.core as og

# og.Controller.edit(
#     {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
#     {
#         og.Controller.Keys.CREATE_NODES: [
#             ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
#             ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
#             ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
#             ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
#             ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
#         ],
#         og.Controller.Keys.CONNECT: [
#             ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
#             ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
#             ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),

#             ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),

#             ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
#             ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
#             ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
#             ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
#         ],
#         og.Controller.Keys.SET_VALUES: [
#             # Providing path to /panda robot to Articulation Controller node
#             # Providing the robot path is equivalent to setting the targetPrim in Articulation Controller node
#             # ("ArticulationController.inputs:usePath", True),      # if you are using an older version of Isaac Sim, you can  uncomment this line
#             ("ArticulationController.inputs:robotPath", "/panda"),
#             ("PublishJointState.inputs:targetPrim", "/panda")
#         ],
#     },
# )