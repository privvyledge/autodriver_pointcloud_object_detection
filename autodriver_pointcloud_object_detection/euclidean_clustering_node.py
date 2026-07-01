"""
ROS2 node for obstacle detection using Euclidean Clustering on Point Clouds.
Publishes clustered pointcloud, visualization markers, and object array.

Usage:
    sudo apt-get install ros-${ROS_DISTRO}-derived-object-msgs
    python3 -m pip install "numpy==1.26.4" scikit-learn hdbscan "open3d==0.19.0"
    ros2 run pointcloud_obstacle_detection euclidean_clustering_node

"""
import sys
import time
import struct
from typing import Union
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy import qos
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2
from sensor_msgs_py.point_cloud2 import read_points, create_cloud
from image_geometry import PinholeCameraModel
from message_filters import Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
from derived_object_msgs.msg import Object, ObjectArray
from shape_msgs.msg import SolidPrimitive
import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, LookupException, ConnectivityException, \
    ExtrapolationException
import numpy as np
import transforms3d

try:
    from tf_transformations import quaternion_from_matrix
except ImportError:
    def quaternion_from_matrix(matrix):
        """Fallback: 4x4/3x3 rotation matrix -> [x, y, z, w] (tf_transformations absent)."""
        m = np.asarray(matrix, dtype=np.float64)[:3, :3]
        t = np.trace(m)
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        return [x, y, z, w]
from cv_bridge import CvBridge
import cv2

import torch
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c

from autodriver_pointcloud_preprocessor.pointcloud_preprocessor import PointcloudPreprocessorNode
from autodriver_pointcloud_preprocessor.utils import (convert_pointcloud_to_numpy, numpy_struct_to_pointcloud2,
                                                      get_current_time, get_time_difference,
                                                      dict_to_open3d_tensor_pointcloud,
                                                      pointcloud_to_dict, get_pointcloud_metadata,
                                                      check_field, crop_pointcloud,
                                                      extract_rgb_from_pointcloud, get_fields_from_dicts,
                                                      remove_duplicates, rgb_float_to_bytes,
                                                      FIELD_DTYPE_MAP, FIELD_DTYPE_MAP_INV)


# from autodriver_pointcloud_object_detection.pointcloud_clustering import (PointCloudClustering,
#                                                                           OPEN3D_INSTALLED, HDBSCAN_INSTALLED,
#                                                                           SKLEARN_INSTALLED)


def get_unique_labels(labels: torch.Tensor, method: int = 1, dim: int = None) -> Union[torch.Tensor, set]:
    # the methods are ordered in terms of speed, currently tested with CPU tensors only
    # todo: test/time with GPU
    unique_labels, counts = [], None

    if method > 4:
        # raise ValueError('Method argument must be between 1 and 4')
        print('Method argument must be between 1 and 4')

    # Method 1: Converting the tensor to a list then set. This does not return counts
    if method == 1:
        start_time = time.perf_counter()
        unique_labels = set(labels.tolist())
        time_set = time.perf_counter() - start_time

    # Method 2: First transfer to cpu then convert to a numpy array followed by sorting and getting unique elements.
    elif method == 2:
        # This method is the slowest due to (1) transferring to CPU, (2) converting the torch tensor, (3) sorting
        start_time = time.perf_counter()
        labels_np = labels.cpu().numpy()
        unique_labels, counts = np.unique(labels_np,
                                      return_counts=True)  # largest_cluster_label = max(labels, key=lambda l: np.sum(labels == l))
        time_np = time.perf_counter() - start_time

    # Method 2: Sort the tensor then get unique consecutive items. Sorting is the largest bottleneck here
    elif method == 3:
        start_time = time.perf_counter()
        labels_sorted, indices_ = labels.sort()
        unique_labels, counts = torch.unique_consecutive(labels_sorted, return_counts=True)
        time_consec = time.perf_counter() - start_time

    # Method 3: Torch unique. Equivalent to sorting followed by unique consecutive but is slower by about 4x.
    elif method == 4:
        start_time = time.perf_counter()
        unique_labels, counts = torch.unique(labels[labels != -1], return_counts=True, sorted=False)
        time_torch = time.perf_counter() - start_time
    return unique_labels, counts


class ObstacleDetectionNode(PointcloudPreprocessorNode):
    def __init__(self):
        # See PointcloudPreprocessorNode for parameters and variables initialized
        super(ObstacleDetectionNode, self).__init__("obstacle_detection_node", enabled=False,
                                                    parameter_namespace='pointcloud_preprocessor',
                                                    enable_parameter_change_callback=False)

        # self.remove_on_set_parameters_callback(super(ObstacleDetectionNode, self).parameter_change_callback)
        # Declare parameters
        # todo: get common parameter values using the parameter namespace
        # self.declare_parameter(name='input_topic', value="/camera/camera/depth/color/points",
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING))
        # self.declare_parameter(name='output_topic', value="/objects_clusters",
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING))
        # self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
        #         description='',
        #         type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('markers_topic', 'obstacle_markers')
        self.declare_parameter('objects_topic', 'detected_objects')
        self.declare_parameter('marker_lifetime', 0.1)  # seconds
        # self.declare_parameter('queue_size', 1)
        # self.declare_parameter('use_gpu', True)
        # self.declare_parameter('cpu_backend', 'numpy')  # numpy or pytorch
        # self.declare_parameter('robot_frame', 'base_link')
        # self.declare_parameter('static_camera_to_robot_tf', True)
        # self.declare_parameter('transform_timeout', 0.1)

        # self.declare_parameter('crop_to_roi', False)
        # self.declare_parameter('roi_min', [-6.0, -6.0, 0.0])
        # self.declare_parameter('roi_max', [6.0, 6.0, 2.0])
        # self.declare_parameter('voxel_size', 0.05)  # 0.01
        # self.declare_parameter('remove_statistical_outliers', False)
        # self.declare_parameter('estimate_normals', False)
        # self.declare_parameter('remove_ground', False)
        # self.declare_parameter('ground_plane', [0.0, 1.0, 0.0, 0.0])
        # self.declare_parameter('use_height', True)  # if true, remove the ground based on height

        self.declare_parameter('clustering_backend', 'open3d')
        self.declare_parameter('clustering_method', 'dbscan')
        self.declare_parameter("cluster_tolerance", 0.2)  # meters
        self.declare_parameter("min_cluster_size", 100)
        self.declare_parameter("max_cluster_size", 1000)  # values <= 0 means no limit
        self.declare_parameter('cluster_min_height', 0.1)  # min height of cluster
        self.declare_parameter('cluster_max_height', 2.0)  # max height of cluster
        self.declare_parameter('verbose', False)

        self.declare_parameter("generate_bounding_box", True)
        self.declare_parameter("bounding_box_type", "AABB")  # AABB or OBB

        # self.declare_parameter("generate_convex_hull", False)

        self.declare_parameter("project_clusters_to_image", False)
        self.declare_parameter("classify_image_clusters", False)
        self.declare_parameter("get_image_from_pointcloud", False)
        self.declare_parameter(name='input_image_topic', value="/camera/camera/color/image_raw",
                               descriptor=ParameterDescriptor(
                                       description='The input image topic. '
                                                   'Works with all image types: RGB(A), BGR(A), mono8, mono16. '
                                                   'If not set or if no message is received, will try to extract image (RGB) from pointcloud.',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='input_camera_info_topic', value="/camera/camera/color/camera_info",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))

        # Get parameters
        # self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        # self.input_topic = self.get_parameter('input_topic').value
        # self.output_topic = self.get_parameter('output_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.marker_lifetime = self.get_parameter('marker_lifetime').value
        # self.qos = self.get_parameter('qos').get_parameter_value().string_value
        # self.queue_size = self.get_parameter_or('queue_size', 1).value
        # self.use_gpu = self.get_parameter_or('use_gpu', 2).value
        # self.cpu_backend = self.get_parameter('cpu_backend').value
        # self.robot_frame = self.get_parameter('robot_frame').value
        # self.static_camera_to_robot_tf = self.get_parameter('static_camera_to_robot_tf').value
        # self.transform_timeout = self.get_parameter('transform_timeout').value
        # self.crop_to_roi = self.get_parameter('crop_to_roi').value
        # self.roi_min = self.get_parameter('roi_min').value
        # self.roi_max = self.get_parameter('roi_max').value
        # self.voxel_size = self.get_parameter('voxel_size').value
        # self.remove_statistical_outliers = self.get_parameter('remove_statistical_outliers').value
        # self.estimate_normals = self.get_parameter('estimate_normals').value
        # self.remove_ground = self.get_parameter('remove_ground').value
        # self.ground_plane = self.get_parameter('ground_plane').value
        # self.use_height = self.get_parameter('use_height').value
        self.clustering_backend = self.get_parameter('clustering_backend').value
        self.clustering_method = self.get_parameter('clustering_method').value
        self.cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        self.max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value
        self.cluster_min_height = self.get_parameter('cluster_min_height').value
        self.cluster_max_height = self.get_parameter('cluster_max_height').value
        self.verbose = self.get_parameter('verbose').value
        self.generate_bounding_box = self.get_parameter("generate_bounding_box").value
        self.bounding_box_type = self.get_parameter("bounding_box_type").value
        # self.generate_convex_hull = self.get_parameter("generate_convex_hull").value

        self.project_clusters_to_image = self.get_parameter("project_clusters_to_image").value
        self.classify_image_clusters = self.get_parameter("classify_image_clusters").value
        self.get_image_from_pointcloud = self.get_parameter("get_image_from_pointcloud").value
        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.input_camera_info_topic = self.get_parameter('input_camera_info_topic').value

        # # Setup the device
        # self.torch_device = torch.device('cpu')
        # self.o3d_device = o3d.core.Device('CPU:0')
        # if self.use_gpu:
        #     if torch.cuda.is_available():
        #         self.torch_device = torch.device('cuda:0')
        #         if o3d.core.cuda.is_available():
        #             self.o3d_device = o3d.core.Device('CUDA:0')

        # Initialize variables
        self.camera_to_robot_tf = None

        # # TF2 broadcaster
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        #
        # # Initialize TF buffer and listener
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        #
        # self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)
        #
        # # Debugging parameters
        # self.processing_times = {}

        # setup QoS
        qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.queue_size
        )
        if self.qos.lower() == "sensor_data":
            qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=self.queue_size
            )

        # Setup dynamic parameter reconfiguring.
        # Register a callback function that will be called whenever there is an attempt to
        # change one or more parameters of the node.
        self.add_on_set_parameters_callback(self.clustering_parameter_change_callback)

        # # #################### Todo: pointcloud clustering test
        # self.clusterer_open3d = PointCloudClustering(method=self.clustering_method, backend=self.clustering_backend)
        # self.clusterer_sklearn_hdbscan = PointCloudClustering(method='hdbscan', backend='sklearn')
        # self.clusterer_sklearn_dbscan = PointCloudClustering(method='dbscan', backend='sklearn')
        # self.clusterer_hdbscan_hdbscan = PointCloudClustering(method='hdbscan', backend='hdbscan')
        # # #################### Todo: pointcloud clustering test

        # Setup subscribers
        # NOTE: subscribe to the RAW LiDAR/depth cloud. This node runs the inherited
        # preprocess() itself, so pointing input_topic at an already-preprocessed topic
        # double-processes it (voxel/ROI/ground/TF run twice). If you must subscribe to a
        # preprocessed topic, disable the inherited steps via the pointcloud_preprocessor
        # namespace: voxel_size:=0.0, crop_to_roi:=false, remove_ground:=false,
        # remove_duplicates:=false.
        self.poincloud_sub = self.create_subscription(PointCloud2, self.input_topic,
                                                      self.callback, qos_profile=qos_profile)

        # Setup publishers
        self.pointcloud_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.objects_pub = self.create_publisher(ObjectArray, self.objects_topic, 10)

        # self.pointcloud_timer = self.create_timer(1 / self.odom_rate, self.rgbd_timer_callback)
        self.get_logger().info(f"{self.get_fully_qualified_name()} node started on device: {self.o3d_device}")


    def callback(self, ros_cloud):
        frame_id = ros_cloud.header.frame_id

        try:
            callback_start_time = get_current_time(monotonic=False)
            if self.extract_pointcloud(ros_cloud) is None:
                return

            # preprocess the PointCloud
            preprocessing_start_time = get_current_time(monotonic=False)
            self.preprocess()
            self.processing_times['preprocessing_time'] = get_time_difference(preprocessing_start_time,
                                                                              get_current_time(monotonic=False))
            # Detect obstacles
            start_time = get_current_time(monotonic=False)
            # Detect obstacles and get clusters
            o3d_labels = self.detect_obstacles()
            self.processing_times['detect_obstacles'] = get_time_difference(start_time,
                                                                            get_current_time(monotonic=False))
            if o3d_labels is None:
                return

            new_header = self.create_header(ros_cloud)
            frame_id = self.robot_frame if self.camera_to_robot_tf is not None else ros_cloud.header.frame_id

            # Get clusters then combine them to form a clustered pointcloud, Get bounding boxes. Create MarkerArray and ObjectArray
            start_time = get_current_time(monotonic=False)
            clusters, bboxes, quat, combined_pcd, marker_array, object_array = self.get_clusters(o3d_labels,
                                                                                                stamp=new_header.stamp,
                                                                                                frame_id=new_header.frame_id)
            self.processing_times['obstacle_postprocessing'] = get_time_difference(start_time,
                                                                            get_current_time(monotonic=False))
            # self.get_logger().info(f"Cluster {clusters}, length: {len(clusters)}, bbox length: {len(bboxes)}")

            if clusters:
                # Publish clustered pointcloud
                if combined_pcd is not None:
                    # Publish processed point cloud
                    start_time = get_current_time(monotonic=True)
                    processed_struct = self.prepare_pointcloud(ros_cloud, o3d_pointcloud=combined_pcd)
                    pc_msg = self.tensor_to_ros_cloud(processed_struct, self.pointfields, header=new_header)
                    pc_msg.is_dense = self.remove_nans and self.remove_infs
                    self.processing_times['prepare_pointcloud'] = get_time_difference(
                        start_time, get_current_time(monotonic=True))
                    start_time = get_current_time(monotonic=True)
                    self.pointcloud_pub.publish(pc_msg)
                    self.processing_times['pointcloud_pub'] = get_time_difference(start_time,
                                                                                  get_current_time(monotonic=True))

                    pcd_number = str(self.frame_count).zfill(8)
                    self.pointcloud_saver(pcd_number)
                    self.pointcloud_visualizer(pcd_number)

                    self.frame_count += 1

                # Create and publish markers
                if marker_array:
                    self.marker_pub.publish(marker_array)

                # Create and publish object array
                if object_array:
                    self.objects_pub.publish(object_array)

            self.processing_times['total_callback_time'] = get_time_difference(callback_start_time, get_current_time(monotonic=False))
            # # Log processing info
            # self.get_logger().info(
            #         f"Published processed pointcloud with "
            #         f"{self.o3d_pointcloud.point.positions.shape[0]} points"
            # )

            # self.get_logger().info(
            #         f"\n Ros to numpy: {1 / self.processing_times['ros_to_numpy']}, "
            #         f"data preparation: {1 / self.processing_times['data_preparation']}, "
            #         f"pcd creation: {1 / self.processing_times['pcd_creation']}, "
            #         f"tensor transfer: {1 / self.processing_times['tensor_transfer']}, "
            #         f"tf_lookup: {1 / self.processing_times['tf_lookup']}, "
            #         f"tf transform: {1 / self.processing_times['transform']}, "
            #         f"crop: {1 / self.processing_times['crop']}, "
            #         f"voxel downsampling: {1 / self.processing_times['voxel_downsampling']}, "
            #         f"statistical outlier removal: {1 / self.processing_times['remove_statistical_outliers']}, "
            #         f"normal estimation: {1 / self.processing_times['normal_estimation']}, "
            #         # f"ground segmentation: {1 / self.processing_times['ground_segmentation']}, "
            #         f"pointcloud parsing: {1 / self.processing_times['pointcloud_msg_parsing']}, "
            #         f"pointcloud publishing: {1 / self.processing_times['pointcloud_pub']} \n"
            # )
        except Exception as e:
            import traceback
            self.get_logger().error(
                f"Error processing point cloud: {str(e)}\n{traceback.format_exc()}")

    def detect_obstacles(self):
        """
        Perform DBSCAN clustering and return clusters with their bounding boxes.
        Return the list of clusters.
        """
        # self.get_logger().info("Clustering point cloud with DBSCAN...")

        # Ensure the point cloud is in memory
        if self.o3d_pointcloud.is_empty():
            return None


        # ############# Todo: test various clustering algos

        # ############# Todo: test various clustering algos

        # Open3D cluster_dbscan is CPU-only even on the tensor API: a GPU cloud is copied
        # to host, clustered, and labels copied back. Keep the post-voxel/ROI point count
        # low (~<5k on Orin Nano) via voxel_size + crop_to_roi to stay real-time.
        if self.verbose:
            self.get_logger().info(
                f"DBSCAN input points: {self.o3d_pointcloud.point.positions.shape[0]}")

        # Use GPU DBSCAN clustering.
        labels = self.o3d_pointcloud.cluster_dbscan(
                eps=self.cluster_tolerance,
                min_points=self.min_cluster_size,
                print_progress=self.verbose
        )
        self.o3d_pointcloud.point.labels = labels
        return labels

    def get_clusters(self, labels: o3c.Tensor, stamp=None, frame_id='', create_objects=True, visualize=True):
        # Convert to a torch tensor without copying
        if isinstance(labels, o3c.Tensor):
            labels = torch.utils.dlpack.from_dlpack(labels.to_dlpack())
        elif isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        # Get unique labels (excluding noise points labeled as -1)
        labels_ = labels[labels != -1]
        unique_labels, counts = get_unique_labels(labels_, method=1)
        unique_labels = sorted(unique_labels)  # deterministic order across frames

        if len(unique_labels) == 0:
            # self.get_logger().info(f"unique labels: {unique_labels}, len: {len(unique_labels)}")
            return [], [], [], [], [], []   # Return empty lists if no valid clusters

        max_label = labels.max().item()
        # self.get_logger().info(f"DBSCAN found {max_label + 1} clusters")

        if self.max_cluster_size > 0 and counts is not None:
            counts_less_than_cluster_size = counts < self.max_cluster_size
            unique_labels = unique_labels[counts_less_than_cluster_size]
            # counts = counts[counts_less_than_cluster_size]

        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        clusters = []
        bboxes = []
        combined_pcd = None
        marker_array, object_array = [], []
        quat = [0.0, 0., 0., 0.]

        if visualize:
            marker_array = MarkerArray()  # Create a MarkerArray message from the detected clusters for visualization.

        if create_objects:
            object_array = ObjectArray()  # Create an ObjectArray message from the detected clusters.
            object_array.header.frame_id = frame_id
            object_array.header.stamp = stamp

        # Group points by label in a single pass instead of an O(N*K) per-cluster boolean
        # mask. argsort once, then each cluster is a contiguous slice of point indices.
        # int64 indices are DLPack-friendly (zero-copy), avoiding the uint8<->bool round-trip
        # and the per-cluster CUDA kernel launches that dominate on the Orin Nano.
        order = torch.argsort(labels)
        sorted_labels = labels[order]
        targets = torch.as_tensor(list(unique_labels), dtype=sorted_labels.dtype,
                                  device=sorted_labels.device)
        left = torch.searchsorted(sorted_labels, targets, right=False)
        right = torch.searchsorted(sorted_labels, targets, right=True)

        # loop through the labels to generate clusters, bboxes
        for i, label in enumerate(unique_labels):
            # Contiguous slice of original point indices belonging to this cluster
            cluster_idx = order[left[i]:right[i]].to(dtype=torch.int64).contiguous()
            cluster_idx = o3c.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(cluster_idx))

            # Create a new pointcloud for the cluster
            cluster_pcd = self.o3d_pointcloud.select_by_index(cluster_idx)

            if cluster_pcd.is_empty():
                continue

            points = cluster_pcd.point.positions

            # Per-cluster size check — primary guard when counts unavailable (e.g. method=1)
            if 0 < self.max_cluster_size < points.shape[0]:
                continue

            # Get cluster height
            min_z = points[:, 2].min().item()
            max_z = points[:, 2].max().item()
            height = max_z - min_z

            # Filter clusters by height. height < self.cluster_min_height or height > self.cluster_max_height
            if not (self.cluster_min_height <= height <= self.cluster_max_height):
                continue

            start_time = get_current_time(monotonic=True)
            # add to combined PCD
            if combined_pcd is None:
                combined_pcd = cluster_pcd
            else:
                combined_pcd = combined_pcd + cluster_pcd
            # append clusters
            clusters.append(cluster_pcd)
            self.processing_times['combine_clusters'] = get_time_difference(start_time, get_current_time(monotonic=True))

            # Get bounding boxes and append
            if self.generate_bounding_box:
                if self.bounding_box_type.lower() == "aabb":
                    bounding_box = cluster_pcd.get_axis_aligned_bounding_box()
                    center = bounding_box.get_center().cpu().numpy().tolist()
                    extent = bounding_box.get_extent().cpu().numpy().tolist()
                    quat = [0.0, 0.0, 0.0, 1.0]  # identity quaternion (x, y, z, w)
                elif self.bounding_box_type.lower() == "obb":
                    bounding_box = cluster_pcd.get_oriented_bounding_box()
                    center = bounding_box.center.cpu().numpy().tolist()
                    extent = bounding_box.extent.cpu().numpy().tolist()
                    # Convert rotation matrix to quaternion
                    R = bounding_box.rotation.cpu().numpy()
                    quat = quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))
                else:
                    raise ValueError(f"Unknown bounding box type: {self.bounding_box_type}")

                bboxes.append(bounding_box)

                # Create marker
                if visualize:
                    marker_msg = self.create_marker(i, center, extent, quat, stamp, frame_id)
                    marker_array.markers.append(marker_msg)

                if create_objects:
                    object_msg = self.create_object(i, center, extent, quat, stamp, frame_id)
                    object_array.objects.append(object_msg)

        return clusters, bboxes, quat, combined_pcd, marker_array, object_array


    def create_marker(self, i, center, extent, quat, stamp, frame_id=''):
        """Create a Marker message from the detected cluster to be appended to a MarkerArray."""

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = "obstacles"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.scale.x = extent[0]
        marker.scale.y = extent[1]
        marker.scale.z = extent[2]

        # Set marker color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        marker.lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime).to_msg()
        return marker

    def create_object(self, i, center, extent, quat, stamp=None, frame_id=''):
        """Create an Object message from the detected cluster to be appended to an ObjectArray.."""
        obj = Object()
        obj.detection_level = Object.OBJECT_DETECTED
        obj.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

        obj.pose.position.x = center[0]
        obj.pose.position.y = center[1]
        obj.pose.position.z = center[2]

        obj.shape.dimensions = [extent[0], extent[1], extent[2]]

        # Optionally add a class label if classification is implemented
        obj.shape.type = SolidPrimitive().BOX
        obj.id = i
        obj.classification = Object.CLASSIFICATION_UNKNOWN
        obj.classification_certainty = int(125)
        return obj


    def clustering_parameter_change_callback(self, params):
        result = SetParametersResult()
        result.successful = True

        result_ = super().parameter_change_callback(params)
        result.reason = result_.reason

        # Iterate over each parameter in this node
        for param in params:
            if param.name == 'cluster_tolerance' and param.type_ == Parameter.Type.DOUBLE:
                self.cluster_tolerance = param.value
            elif param.name == 'min_cluster_size' and param.type_ == Parameter.Type.INTEGER:
                self.min_cluster_size = param.value
            elif param.name == 'max_cluster_size' and param.type_ == Parameter.Type.INTEGER:
                self.max_cluster_size = param.value
            elif param.name == 'cluster_min_height' and param.type_ == Parameter.Type.DOUBLE:
                self.cluster_min_height = param.value
            elif param.name == 'cluster_max_height' and param.type_ == Parameter.Type.DOUBLE:
                self.cluster_max_height = param.value
            elif param.name == 'generate_bounding_box' and param.type_ == Parameter.Type.BOOL:
                self.generate_bounding_box = param.value
            elif param.name == 'bounding_box_type' and param.type_ == Parameter.Type.STRING:
                self.bounding_box_type = param.value
            elif param.name == 'project_clusters_to_image' and param.type_ == Parameter.Type.BOOL:
                self.project_clusters_to_image = param.value
            elif param.name == 'classify_image_clusters' and param.type_ == Parameter.Type.BOOL:
                self.classify_image_clusters = param.value
            else:
                result.successful = result_.successful or False
            self.get_logger().info(f"Success = {result.successful} for param {param.name} to value {param.value}")
        return result


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info("Shutting down node...")
    finally:
        if node.visualize:
            node.vis.destroy_window()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
