import omni.graph.core as og

ROS_CAMERA_GRAPH_PATH = "/Camera_Publish"
# FORK_HEEL_CAMERA_PATH = "/World/lola/fork/camera/camera"
FORK_CAMERA_LEFT_PATH = "/World/lola/fork/cameras/camera_left/camera"
FORK_CAMERA_RIGHT_PATH = "/World/lola/fork/cameras/camera_right/camera"
CAMERA_RESOLUTION = (960, 600)
# FORKHEEL_CAMERA_NAME = "forkheel_camera"
FORK_CAMERA_LEFT_NAME = "fork_camera_left"
FORK_CAEMRA_RIGHT_NAME = "fork_camera_right"

# Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers
keys = og.Controller.Keys
(camera_publish_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": ROS_CAMERA_GRAPH_PATH,
        "evaluator_name": "execution",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND
    },
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),

            # Left Camera
            ("RenderProduct_Left", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraHelperInfo_Left", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ("CameraHelperRgb_Left", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperDepth_Left", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            # ("CameraHelperSemantic", "isaacsim.ros2.bridge.ROS2CameraHelper"),

            # Right Camera
            ("RenderProduct_Right", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraHelperInfo_Right", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ("CameraHelperRgb_Right", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperDepth_Right", "isaacsim.ros2.bridge.ROS2CameraHelper")
            
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "RenderProduct_Left.inputs:execIn"),
            ("OnPlaybackTick.outputs:tick", "RenderProduct_Right.inputs:execIn"),

            ("RenderProduct_Left.outputs:execOut", "CameraHelperInfo_Left.inputs:execIn"),
            ("RenderProduct_Left.outputs:execOut", "CameraHelperRgb_Left.inputs:execIn"),
            ("RenderProduct_Left.outputs:execOut", "CameraHelperDepth_Left.inputs:execIn"),

            ("RenderProduct_Right.outputs:execOut", "CameraHelperInfo_Right.inputs:execIn"),
            ("RenderProduct_Right.outputs:execOut", "CameraHelperRgb_Right.inputs:execIn"),
            ("RenderProduct_Right.outputs:execOut", "CameraHelperDepth_Right.inputs:execIn"),

            # ("RenderProduct_Left.outputs:execOut", "CameraHelperSemantic.inputs:execIn"),
            ("RenderProduct_Left.outputs:renderProductPath", "CameraHelperInfo_Left.inputs:renderProductPath"),
            ("RenderProduct_Left.outputs:renderProductPath", "CameraHelperRgb_Left.inputs:renderProductPath"),
            ("RenderProduct_Left.outputs:renderProductPath", "CameraHelperDepth_Left.inputs:renderProductPath"),
            # ("RenderProduct_Left.outputs:renderProductPath", "CameraHelperSemantic.inputs:renderProductPath")

            ("RenderProduct_Right.outputs:renderProductPath", "CameraHelperInfo_Right.inputs:renderProductPath"),
            ("RenderProduct_Right.outputs:renderProductPath", "CameraHelperRgb_Right.inputs:renderProductPath"),
            ("RenderProduct_Right.outputs:renderProductPath", "CameraHelperDepth_Right.inputs:renderProductPath"),
        ],
        keys.SET_VALUES: [
            ("CameraHelperInfo_Left.inputs:frameId", FORK_CAMERA_LEFT_NAME),
            ("CameraHelperInfo_Left.inputs:topicName", f"{FORK_CAMERA_LEFT_NAME}/info"),
            ("CameraHelperRgb_Left.inputs:frameId", FORK_CAMERA_LEFT_NAME),
            ("CameraHelperRgb_Left.inputs:topicName", f"{FORK_CAMERA_LEFT_NAME}/rgb"),
            ("CameraHelperRgb_Left.inputs:type", "rgb"),
            ("CameraHelperDepth_Left.inputs:frameId", FORK_CAMERA_LEFT_NAME),
            ("CameraHelperDepth_Left.inputs:topicName", f"{FORK_CAMERA_LEFT_NAME}/depth"),
            ("CameraHelperDepth_Left.inputs:type", "depth"),

             ("CameraHelperInfo_Right.inputs:frameId", FORK_CAEMRA_RIGHT_NAME),
            ("CameraHelperInfo_Right.inputs:topicName", f"{FORK_CAEMRA_RIGHT_NAME}/info"),
            ("CameraHelperRgb_Right.inputs:frameId", FORK_CAEMRA_RIGHT_NAME),
            ("CameraHelperRgb_Right.inputs:topicName", f"{FORK_CAEMRA_RIGHT_NAME}/rgb"),
            ("CameraHelperRgb_Right.inputs:type", "rgb"),
            ("CameraHelperDepth_Right.inputs:frameId", FORK_CAEMRA_RIGHT_NAME),
            ("CameraHelperDepth_Right.inputs:topicName", f"{FORK_CAEMRA_RIGHT_NAME}/depth"),
            ("CameraHelperDepth_Right.inputs:type", "depth"),
            # ("CameraHelperSemantic.inputs:frameId", FORKHEEL_CAMERA_NAME),
            # ("CameraHelperSemantic.inputs:topicName", f"{FORKHEEL_CAMERA_NAME}/semantic_segmentation"),
            # ("CameraHelperSemantic.inputs:type", "semantic_segmentation"),
            # ("CameraHelperSemantic.inputs:enableSemanticLabels", True),

            ("RenderProduct_Left.inputs:cameraPrim", FORK_CAMERA_LEFT_PATH),
            ("RenderProduct_Left.inputs:width", CAMERA_RESOLUTION[0]),
            ("RenderProduct_Left.inputs:height", CAMERA_RESOLUTION[1]),

            ("RenderProduct_Right.inputs:cameraPrim", FORK_CAMERA_RIGHT_PATH),
            ("RenderProduct_Right.inputs:width", CAMERA_RESOLUTION[0]),
            ("RenderProduct_Right.inputs:height", CAMERA_RESOLUTION[1])
        ]
    }
)
