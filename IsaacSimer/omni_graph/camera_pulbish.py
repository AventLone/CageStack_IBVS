import omni.graph.core as og

ROS_CAMERA_GRAPH_PATH = "/Camera_Publish"
MID_CAMERA_PRIM_PATH = "/World/lola/side_shifter/camera/camera"
LEFT_CAMERA_PRIM_PATH = "/World/lola/fork/cameras/camera_left/camera"
RIGHT_CAMERA_PRIM_PATH = "/World/lola/fork/cameras/camera_right/camera"
CAMERA_RESOLUTION = (1920, 1200)

# Creating an on-demand push graph with cameraHelper nodes to generate ROS image publishers
keys = og.Controller.Keys
(camera_publish_graph, _, _, _) = og.Controller.edit(
    {
        "graph_path": ROS_CAMERA_GRAPH_PATH,
        "evaluator_name": "push",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND
    },
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),

            # Camera on sideshifter
            # ("RenderProduct_sideshifter", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            # ("CameraHelperInfo_sideshifter", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            # ("CameraHelperRgb_sideshifter", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            # ("CameraHelperDepth_sideshifter", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            # ("CameraHelperInstance_sideshifter", "isaacsim.ros2.bridge.ROS2CameraHelper"),

            # Camera on left fork tip
            ("RenderProduct_left", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraHelperInfo_left", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ("CameraHelperRgb_left", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperDepth_left", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperInstance_left", "isaacsim.ros2.bridge.ROS2CameraHelper"),

            # Camera on right fork tip
            ("RenderProduct_right", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            ("CameraHelperInfo_right", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ("CameraHelperRgb_right", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperDepth_right", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ("CameraHelperInstance_right", "isaacsim.ros2.bridge.ROS2CameraHelper")
        ],
        keys.CONNECT: [
            # Mid Camera
            # ("OnPlaybackTick.outputs:tick", "RenderProduct_sideshifter.inputs:execIn"),
            # ("RenderProduct_sideshifter.outputs:execOut", "CameraHelperInfo_sideshifter.inputs:execIn"),
            # ("RenderProduct_sideshifter.outputs:execOut", "CameraHelperRgb_sideshifter.inputs:execIn"),
            # ("RenderProduct_sideshifter.outputs:execOut", "CameraHelperDepth_sideshifter.inputs:execIn"),
            # ("RenderProduct_sideshifter.outputs:execOut", "CameraHelperInstance_sideshifter.inputs:execIn"),
            # ("RenderProduct_sideshifter.outputs:renderProductPath", "CameraHelperInfo_sideshifter.inputs:renderProductPath"),
            # ("RenderProduct_sideshifter.outputs:renderProductPath", "CameraHelperRgb_sideshifter.inputs:renderProductPath"),
            # ("RenderProduct_sideshifter.outputs:renderProductPath", "CameraHelperDepth_sideshifter.inputs:renderProductPath"),
            # ("RenderProduct_sideshifter.outputs:renderProductPath", "CameraHelperInstance_sideshifter.inputs:renderProductPath"),

            # Left Camera
            ("OnPlaybackTick.outputs:tick", "RenderProduct_left.inputs:execIn"),
            ("RenderProduct_left.outputs:execOut", "CameraHelperInfo_left.inputs:execIn"),
            ("RenderProduct_left.outputs:execOut", "CameraHelperRgb_left.inputs:execIn"),
            ("RenderProduct_left.outputs:execOut", "CameraHelperDepth_left.inputs:execIn"),
            ("RenderProduct_left.outputs:execOut", "CameraHelperInstance_left.inputs:execIn"),
            ("RenderProduct_left.outputs:renderProductPath", "CameraHelperInfo_left.inputs:renderProductPath"),
            ("RenderProduct_left.outputs:renderProductPath", "CameraHelperRgb_left.inputs:renderProductPath"),
            ("RenderProduct_left.outputs:renderProductPath", "CameraHelperDepth_left.inputs:renderProductPath"),
            ("RenderProduct_left.outputs:renderProductPath", "CameraHelperInstance_left.inputs:renderProductPath"),

            # Right Camera
            ("OnPlaybackTick.outputs:tick", "RenderProduct_right.inputs:execIn"),
            ("RenderProduct_right.outputs:execOut", "CameraHelperInfo_right.inputs:execIn"),
            ("RenderProduct_right.outputs:execOut", "CameraHelperRgb_right.inputs:execIn"),
            ("RenderProduct_right.outputs:execOut", "CameraHelperDepth_right.inputs:execIn"),
            ("RenderProduct_right.outputs:execOut", "CameraHelperInstance_right.inputs:execIn"),
            ("RenderProduct_right.outputs:renderProductPath", "CameraHelperInfo_right.inputs:renderProductPath"),
            ("RenderProduct_right.outputs:renderProductPath", "CameraHelperRgb_right.inputs:renderProductPath"),
            ("RenderProduct_right.outputs:renderProductPath", "CameraHelperDepth_right.inputs:renderProductPath"),
            ("RenderProduct_right.outputs:renderProductPath", "CameraHelperInstance_right.inputs:renderProductPath")
        ],
        keys.SET_VALUES: [
            # ("CameraHelperInfo_sideshifter.inputs:frameId", "mid_camera"),
            # ("CameraHelperInfo_sideshifter.inputs:topicName", "mid_camera/info"),
            # ("CameraHelperRgb_sideshifter.inputs:frameId", "mid_camera"),
            # ("CameraHelperRgb_sideshifter.inputs:topicName", "mid_camera/rgb"),
            # ("CameraHelperRgb_sideshifter.inputs:type", "rgb"),
            # ("CameraHelperDepth_sideshifter.inputs:frameId", "mid_camera"),
            # ("CameraHelperDepth_sideshifter.inputs:topicName", "mid_camera/depth"),
            # ("CameraHelperDepth_sideshifter.inputs:type", "depth"),
            # ("CameraHelperInstance_sideshifter.inputs:frameId", "mid_camera"),
            # ("CameraHelperInstance_sideshifter.inputs:topicName", "mid_camera/instance_segmentation"),
            # ("CameraHelperInstance_sideshifter.inputs:type", "instance_segmentation"),
            # ("CameraHelperInstance_sideshifter.inputs:enableSemanticLabels", True),
            
            ("CameraHelperInfo_left.inputs:frameId", "left_camera"),
            ("CameraHelperInfo_left.inputs:topicName", "left_camera/info"),
            ("CameraHelperRgb_left.inputs:frameId", "left_camera"),
            ("CameraHelperRgb_left.inputs:topicName", "left_camera/rgb"),
            ("CameraHelperRgb_left.inputs:type", "rgb"),
            ("CameraHelperDepth_left.inputs:frameId", "left_camera"),
            ("CameraHelperDepth_left.inputs:topicName", "left_camera/depth"),
            ("CameraHelperDepth_left.inputs:type", "depth"),
            ("CameraHelperInstance_left.inputs:frameId", "left_camera"),
            ("CameraHelperInstance_left.inputs:topicName", "left_camera/instance_segmentation"),
            ("CameraHelperInstance_left.inputs:type", "instance_segmentation"),
            ("CameraHelperInstance_left.inputs:enableSemanticLabels", True),

            ("CameraHelperInfo_right.inputs:frameId", "right_camera"),
            ("CameraHelperInfo_right.inputs:topicName", "right_camera/info"),
            ("CameraHelperRgb_right.inputs:frameId", "right_camera"),
            ("CameraHelperRgb_right.inputs:topicName", "right_camera/rgb"),
            ("CameraHelperRgb_right.inputs:type", "rgb"),
            ("CameraHelperDepth_right.inputs:frameId", "right_camera"),
            ("CameraHelperDepth_right.inputs:topicName", "right_camera/depth"),
            ("CameraHelperDepth_right.inputs:type", "depth"),
            ("CameraHelperInstance_right.inputs:frameId", "right_camera"),
            ("CameraHelperInstance_right.inputs:topicName", "right_camera/instance_segmentation"),
            ("CameraHelperInstance_right.inputs:type", "instance_segmentation"),
            ("CameraHelperInstance_right.inputs:enableSemanticLabels", True),

            # ("RenderProduct_sideshifter.inputs:cameraPrim", MID_CAMERA_PRIM_PATH),
            ("RenderProduct_left.inputs:cameraPrim", LEFT_CAMERA_PRIM_PATH),
            ("RenderProduct_left.inputs:width", CAMERA_RESOLUTION[0]),
            ("RenderProduct_left.inputs:height", CAMERA_RESOLUTION[1]),

            ("RenderProduct_right.inputs:cameraPrim", RIGHT_CAMERA_PRIM_PATH),
            ("RenderProduct_right.inputs:width", CAMERA_RESOLUTION[0]),
            ("RenderProduct_right.inputs:height", CAMERA_RESOLUTION[1])
        ]
    }
)