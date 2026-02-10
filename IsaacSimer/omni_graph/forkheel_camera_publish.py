import omni.graph.core as og

ROS_CAMERA_GRAPH_PATH = "/Camera_Publish"
FORK_HEEL_CAMERA_PATH = "/World/lola/fork/camera/camera"
CAMERA_RESOLUTION = (1920, 1200)
FORKHEEL_CAMERA_NAME = "forkheel_camera"

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
            ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperSemantic", "isaacsim.ros2.bridge.ROS2CameraHelper")
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick", "RenderProduct.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "CameraHelperInfo.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "CameraHelperRgb.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "CameraHelperDepth.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "CameraHelperSemantic.inputs:execIn"),
            ("RenderProduct.outputs:renderProductPath", "CameraHelperInfo.inputs:renderProductPath"),
            ("RenderProduct.outputs:renderProductPath", "CameraHelperRgb.inputs:renderProductPath"),
            ("RenderProduct.outputs:renderProductPath", "CameraHelperDepth.inputs:renderProductPath"),
            ("RenderProduct.outputs:renderProductPath", "CameraHelperSemantic.inputs:renderProductPath")
        ],
        keys.SET_VALUES: [
            ("CameraHelperInfo.inputs:frameId", FORKHEEL_CAMERA_NAME),
            ("CameraHelperInfo.inputs:topicName", f"{FORKHEEL_CAMERA_NAME}/info"),
            ("CameraHelperRgb.inputs:frameId", FORKHEEL_CAMERA_NAME),
            ("CameraHelperRgb.inputs:topicName", f"{FORKHEEL_CAMERA_NAME}/rgb"),
            ("CameraHelperRgb.inputs:type", "rgb"),
            ("CameraHelperDepth.inputs:frameId", FORKHEEL_CAMERA_NAME),
            ("CameraHelperDepth.inputs:topicName", f"{FORKHEEL_CAMERA_NAME}/depth"),
            ("CameraHelperDepth.inputs:type", "depth"),
            ("CameraHelperSemantic.inputs:frameId", FORKHEEL_CAMERA_NAME),
            ("CameraHelperSemantic.inputs:topicName", f"{FORKHEEL_CAMERA_NAME}/semantic_segmentation"),
            ("CameraHelperSemantic.inputs:type", "semantic_segmentation"),
            ("CameraHelperSemantic.inputs:enableSemanticLabels", True),

            ("RenderProduct.inputs:cameraPrim", FORK_HEEL_CAMERA_PATH),
            ("RenderProduct.inputs:width", CAMERA_RESOLUTION[0]),
            ("RenderProduct.inputs:height", CAMERA_RESOLUTION[1])
        ]
    }
)
