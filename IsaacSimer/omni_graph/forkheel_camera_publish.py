import omni.graph.core as og

ROS_CAMERA_GRAPH_PATH = "/Camera_Publish"
FORK_HEEL_CAMERA_PATH = "/World/lola/fork/camera/camera"
CAMERA_RESOLUTION = (1920, 1200)

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
            # Mid Camera
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
            ("CameraHelperInfo.inputs:frameId", "forkheel_camera"),
            ("CameraHelperInfo.inputs:topicName", "forkheel_camera/info"),
            ("CameraHelperRgb.inputs:frameId", "forkheel_camera"),
            ("CameraHelperRgb.inputs:topicName", "forkheel_camera/rgb"),
            ("CameraHelperRgb.inputs:type", "rgb"),
            ("CameraHelperDepth.inputs:frameId", "forkheel_camera"),
            ("CameraHelperDepth.inputs:topicName", "forkheel_camera/depth"),
            ("CameraHelperDepth.inputs:type", "depth"),
            ("CameraHelperSemantic.inputs:frameId", "forkheel_camera"),
            ("CameraHelperSemantic.inputs:topicName", "forkheel_camera/semantic_segmentation"),
            ("CameraHelperSemantic.inputs:type", "semantic_segmentation"),
            ("CameraHelperSemantic.inputs:enableSemanticLabels", True),

            ("RenderProduct.inputs:cameraPrim", FORK_HEEL_CAMERA_PATH),
            ("RenderProduct.inputs:width", CAMERA_RESOLUTION[0]),
            ("RenderProduct.inputs:height", CAMERA_RESOLUTION[1])
        ]
    }
)
