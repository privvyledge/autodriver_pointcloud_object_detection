"""
ROS2 node for obstacle detection using Euclidean Clustering on Point Clouds.
Publishes clustered pointcloud, visualization markers, and object array.

Usage:
    sudo apt-get install ros-${ROS_DISTRO}-derived-object-msgs
    ros2 run pointcloud_obstacle_detection euclidean_clustering_node

Todo:
    * Use python composition to run this node with pointcloud preprocessor
"""

import time
import struct
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy import qos
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
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
import tf_transformations
import numpy as np
import transforms3d
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from cv_bridge import CvBridge
import cv2
import torch
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c


class ObstacleDetectionNode(Node):
    def __init__(self):
        super(ObstacleDetectionNode, self).__init__("obstacle_detection_node")

        # Declare parameters
        self.declare_parameter(name='input_topic', value="/camera/camera/depth/color/points",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='output_topic', value="/objects_clusters",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('markers_topic', '/obstacle_markers')
        self.declare_parameter('objects_topic', '/detected_objects')
        self.declare_parameter('marker_lifetime', 0.1)  # seconds
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('cpu_backend', 'numpy')  # numpy or pytorch
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter('transform_timeout', 0.1)

        self.declare_parameter('crop_to_roi', False)
        self.declare_parameter('roi_min', [-6.0, -6.0, 0.0])
        self.declare_parameter('roi_max', [6.0, 6.0, 2.0])
        self.declare_parameter('voxel_size', 0.05)  # 0.01
        self.declare_parameter('remove_statistical_outliers', False)
        self.declare_parameter('estimate_normals', False)
        self.declare_parameter('remove_ground', False)
        self.declare_parameter('ground_plane', [0.0, 1.0, 0.0, 0.0])
        self.declare_parameter('use_height', True)  # if true, remove the ground based on height

        self.declare_parameter("cluster_tolerance", 0.2)  # meters
        self.declare_parameter("min_cluster_size", 100)
        self.declare_parameter("max_cluster_size", 10000)
        self.declare_parameter('cluster_min_height', 0.1)  # min height of cluster
        self.declare_parameter('cluster_max_height', 2.0)  # max height of cluster

        self.declare_parameter("generate_bounding_box", True)
        self.declare_parameter("bounding_box_type", "AABB")  # AABB or OBB

        # self.declare_parameter("generate_convex_hull", False)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.markers_topic = self.get_parameter('markers_topic').value
        self.objects_topic = self.get_parameter('objects_topic').value
        self.marker_lifetime = self.get_parameter('marker_lifetime').value
        self.qos = self.get_parameter('qos').get_parameter_value().string_value
        self.queue_size = self.get_parameter('queue_size').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.cpu_backend = self.get_parameter('cpu_backend').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.static_camera_to_robot_tf = self.get_parameter('static_camera_to_robot_tf').value
        self.transform_timeout = self.get_parameter('transform_timeout').value
        self.crop_to_roi = self.get_parameter('crop_to_roi').value
        self.roi_min = self.get_parameter('roi_min').value
        self.roi_max = self.get_parameter('roi_max').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.remove_statistical_outliers = self.get_parameter('remove_statistical_outliers').value
        self.estimate_normals = self.get_parameter('estimate_normals').value
        self.remove_ground = self.get_parameter('remove_ground').value
        self.ground_plane = self.get_parameter('ground_plane').value
        self.use_height = self.get_parameter('use_height').value
        self.cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        self.max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value
        self.cluster_min_height = self.get_parameter('cluster_min_height').value
        self.cluster_max_height = self.get_parameter('cluster_max_height').value
        self.generate_bounding_box = self.get_parameter("generate_bounding_box").value
        self.bounding_box_type = self.get_parameter("bounding_box_type").value
        # self.generate_convex_hull = self.get_parameter("generate_convex_hull").value

        # Setup the device
        self.torch_device = torch.device('cpu')
        self.o3d_device = o3d.core.Device('CPU:0')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.torch_device = torch.device('cuda:0')
                if o3d.core.cuda.is_available():
                    self.o3d_device = o3d.core.Device('CUDA:0')

        # Initialize variables
        self.camera_to_robot_tf = None

        # TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)

        # Debugging parameters
        self.processing_times = {}

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

        # Setup subscribers
        self.poincloud_sub = self.create_subscription(PointCloud2, self.input_topic,
                                                      self.callback, qos_profile=qos_profile)

        # Setup publishers
        self.pointcloud_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)
        self.marker_pub = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.objects_pub = self.create_publisher(ObjectArray, self.objects_topic, 10)

        # self.pointcloud_timer = self.create_timer(1 / self.odom_rate, self.rgbd_timer_callback)
        self.get_logger().info(f"obstacle_detection_node node started on device: {self.o3d_device}")

    def callback(self, ros_cloud):
        frame_id = ros_cloud.header.frame_id

        try:
            start_time = time.time()
            # Get field indices for faster access
            xyz_offset = [None, None, None]
            rgb_offset = None

            for idx, field in enumerate(ros_cloud.fields):
                if field.name == 'x':
                    xyz_offset[0] = idx
                elif field.name == 'y':
                    xyz_offset[1] = idx
                elif field.name == 'z':
                    xyz_offset[2] = idx
                elif field.name == 'rgb':
                    rgb_offset = idx

            if None in xyz_offset or rgb_offset is None:
                self.get_logger().error("Required point cloud fields not found")
                return

            # Convert ROS PointCloud2 to numpy arrays
            # https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
            # https://gist.github.com/SebastianGrans/6ae5cab66e453a14a859b66cd9579239?permalink_comment_id=4345802#gistcomment-4345802
            cloud_array = point_cloud2.read_points_numpy(
                    ros_cloud,
                    field_names=('x', 'y', 'z', 'rgb'),
                    skip_nans=True
            )

            self.processing_times['ros_to_numpy'] = time.time() - start_time

            # Extract XYZ points
            start_time = time.time()
            points_np = cloud_array[:, :3].astype(np.float32)

            # Extract and convert RGB values
            rgb_float = cloud_array[:, 3].copy()
            rgb_bytes = rgb_float.view(np.uint32)

            # Extract RGB channels
            r = ((rgb_bytes >> 16) & 0xFF).astype(np.float32) / 255.0
            g = ((rgb_bytes >> 8) & 0xFF).astype(np.float32) / 255.0
            b = (rgb_bytes & 0xFF).astype(np.float32) / 255.0

            # Stack RGB channels
            colors_np = np.vstack((r, g, b)).T
            self.processing_times['data_preparation'] = time.time() - start_time

            # Convert numpy arrays to tensors and move to device
            start_time = time.time()
            self.o3d_pointcloud.point.positions = o3d.core.Tensor(
                    points_np,
                    dtype=o3d.core.Dtype.Float32,
                    device=self.o3d_device
            )

            self.o3d_pointcloud.point.colors = o3d.core.Tensor(
                    colors_np,
                    dtype=o3d.core.Dtype.Float32,
                    device=self.o3d_device
            )
            self.processing_times['tensor_transfer'] = time.time() - start_time

            # Preprocess pointcloud
            start_time = time.time()
            self.preprocess_pointcloud(frame_id, ros_cloud.header.stamp)
            self.processing_times['preprocess_pointcloud'] = time.time() - start_time

            # Detect obstacles
            start_time = time.time()
            # Detect obstacles and get clusters
            clusters, bboxes = self.detect_obstacles()
            self.get_logger().info(f"Cluster {clusters}, length: {len(clusters)}, bbox length: {len(bboxes)}")

            if clusters:
                # Publish clustered pointcloud
                clustered_cloud = self.combine_clusters(clusters)

                if clustered_cloud is not None:
                    start_time = time.time()
                    pc_msg = self.tensor_to_ros_cloud(clustered_cloud, frame_id=self.robot_frame)
                    self.pointcloud_pub.publish(pc_msg)
                    self.processing_times['pointcloud_msg_parsing'] = time.time() - start_time

                # Create and publish markers
                marker_array = self.create_marker_array(clusters, bboxes, frame_id=self.robot_frame)
                self.marker_pub.publish(marker_array)

                # Create and publish object array
                object_array = self.create_object_array(clusters, bboxes, frame_id=self.robot_frame)
                self.objects_pub.publish(object_array)

            self.processing_times['detect_obstacles'] = time.time() - start_time

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
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def preprocess_pointcloud(self, frame_id, timestamp):
        # transform to robot frame
        start_time = time.time()
        self.get_camera_to_robot_tf(frame_id, timestamp)
        self.processing_times['tf_lookup'] = time.time() - start_time

        if self.camera_to_robot_tf is not None:
            start_time = time.time()
            self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf)
            frame_id = self.robot_frame
            self.processing_times['transform'] = time.time() - start_time

        ## Remove duplicate points
        #start_time = time.time()
        #mask = self.o3d_pointcloud.remove_duplicated_points()
        #self.o3d_pointcloud = self.o3d_pointcloud.select_by_mask(mask)
        #self.processing_times['remove_duplicate_points'] = time.time() - start_time

        ## Remove NaN points
        #start_time = time.time()
        #self.o3d_pointcloud = self.o3d_pointcloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
        #self.processing_times['remove_nan_points'] = time.time() - start_time

        # ROI cropping
        if self.crop_to_roi:
            start_time = time.time()
            # points_o3d = points_o3d.crop(self.roi_min, self.roi_max)
            mask = (
                    (self.o3d_pointcloud.point.positions[:, 0] >= self.roi_min[0]) &
                    (self.o3d_pointcloud.point.positions[:, 0] <= self.roi_max[0]) &
                    (self.o3d_pointcloud.point.positions[:, 1] >= self.roi_min[1]) &
                    (self.o3d_pointcloud.point.positions[:, 1] <= self.roi_max[1]) &
                    (self.o3d_pointcloud.point.positions[:, 2] >= self.roi_min[2]) &
                    (self.o3d_pointcloud.point.positions[:, 2] <= self.roi_max[2])
            )
            self.o3d_pointcloud = self.o3d_pointcloud.select_by_mask(mask)
            self.processing_times['crop'] = time.time() - start_time

        # Voxel downsampling
        if self.voxel_size > 0.0:
            start_time = time.time()
            self.o3d_pointcloud = self.o3d_pointcloud.voxel_down_sample(self.voxel_size)
            self.processing_times['voxel_downsampling'] = time.time() - start_time

        if self.remove_statistical_outliers:
            start_time = time.time()
            self.o3d_pointcloud, _ = self.o3d_pointcloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
            self.processing_times['remove_statistical_outliers'] = time.time() - start_time

        if self.estimate_normals:
            start_time = time.time()
            self.o3d_pointcloud.estimate_normals(
                    radius=0.1,  # Use a radius of 10 cm for local geometry
                    max_nn=30  # Use up to 30 nearest neighbors
            )
            self.processing_times['normal_estimation'] = time.time() - start_time

        # Ground segmentation.
        if self.remove_ground:
            start_time = time.time()
            plane_model, inliers = self.o3d_pointcloud.segment_plane(
                    distance_threshold=0.2,
                    ransac_n=5,
                    num_iterations=100
            )
            # ground_cloud = self.o3d_pointcloud.select_by_index(inliers)  # ground
            self.o3d_pointcloud = self.o3d_pointcloud.select_by_index(inliers, invert=True)  #
            self.processing_times['ground_segmentation'] = time.time() - start_time

    def get_camera_to_robot_tf(self, source_frame_id, timestamp=None):
        if self.camera_to_robot_tf is not None and self.static_camera_to_robot_tf:
            return

        if timestamp is None:
            timestamp = rclpy.time.Time()
        if self.robot_frame:
            # Try to get the transform from camera to robot
            try:
                transform = self.tf_buffer.lookup_transform(
                        self.robot_frame,
                        source_frame_id,
                        timestamp,  # this could also be the depth msg timestamp
                        rclpy.duration.Duration(seconds=self.transform_timeout)
                )
            except tf2_ros.LookupException as e:
                self.get_logger().error(f"TF Lookup Error: {str(e)}")
                return
            except tf2_ros.ConnectivityException as e:
                self.get_logger().error(f"TF Connectivity Error: {str(e)}")
                return
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().error(f"TF Extrapolation Error: {str(e)}")
                return

            # Convert the TF transform to a 4x4 transformation matrix
            self.camera_to_robot_tf = self.transform_to_matrix(transform)
            return

    def transform_to_matrix(self, transform: TransformStamped):
        """Convert TransformStamped to 4x4 transformation matrix."""
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        matrix[:3, 3] = [translation.x, translation.y, translation.z]

        tf_matrix = o3c.Tensor(matrix, dtype=o3c.float32, device=self.o3d_device)

        # self.camera_to_robot_tf = tf_matrix
        return tf_matrix

    def detect_obstacles(self):
        """
        Perform DBSCAN clustering and return clusters with their bounding boxes.
        Return the list of clusters.
        Todo: remove clusters with more points than self.max_cluster_size
        """
        self.get_logger().info("Clustering point cloud with DBSCAN...")

        # Ensure the point cloud is in GPU memory
        if self.o3d_pointcloud.is_empty():
            self.get_logger().info(f"Pointcloud is empty")
            return [], []

        # Use GPU DBSCAN clustering. Todo: replace numpy with pytorch and keep on device
        labels = self.o3d_pointcloud.cluster_dbscan(
                eps=self.cluster_tolerance,
                min_points=self.min_cluster_size,
                print_progress=True  # todo: set to False
        )

        # Get unique labels (excluding noise points labeled as -1)
        labels = labels.cpu().numpy()  # todo: transfer to torch using dlpack and get unique labels
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            self.get_logger().info(f"unique labels: {unique_labels}, len: {len(unique_labels)}")
            return [], []  # Return empty lists if no valid clusters

        #unique_labels = unique_labels[unique_labels >= 0]

        max_label = labels.max().item()
        self.get_logger().info(f"DBSCAN found {max_label + 1} clusters")

        clusters = []
        bboxes = []

        for label in unique_labels:
            # Create mask for current cluster
            mask = labels == label
            mask = o3c.Tensor(mask, device=self.o3d_device)

            # Create new pointcloud for cluster
            cluster_pcd = self.o3d_pointcloud.select_by_mask(mask)

            # # Get cluster height. todo: use mask
            # points = cluster_pcd.point.positions.cpu().numpy()
            # min_z = np.min(points[:, 2])
            # max_z = np.max(points[:, 2])
            # height = max_z - min_z
            #
            # # Filter clusters by height
            # if height < self.cluster_min_height or height > self.cluster_max_height:
            #     continue

            clusters.append(cluster_pcd)

            if self.generate_bounding_box:
                if self.bounding_box_type.lower() == "aabb":
                    bounding_box = cluster_pcd.get_axis_aligned_bounding_box()
                elif self.bounding_box_type.lower() == "obb":
                    bounding_box = cluster_pcd.get_oriented_bounding_box()
                else:
                    raise ValueError(f"Unknown bounding box type: {self.bounding_box_type}")

                bboxes.append(bounding_box)

        return clusters, bboxes

    def combine_clusters(self, clusters):
        """Combine all clusters into one point cloud."""
        if not clusters:
            return None
        combined_pcd = clusters[0]
        for cluster in clusters[1:]:
            combined_pcd = combined_pcd + cluster
        return combined_pcd

    def tensor_to_ros_cloud(self, pcd_tensor, frame_id="camera_link", stamp=None):
        """Convert Open3D tensor pointcloud to ROS PointCloud2 message"""
        # Get points and colors from tensor (moved to CPU if necessary)
        points = pcd_tensor.point.positions.cpu().numpy()
        colors = pcd_tensor.point.colors.cpu().numpy()

        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        # Convert colors from float [0,1] to uint32 packed RGB
        colors_uint32 = (colors * 255).astype(np.uint8)
        rgb_packed = np.zeros(len(points), dtype=np.uint32)
        rgb_packed = np.left_shift(colors_uint32[:, 0].astype(np.uint32), 16) | \
                    np.left_shift(colors_uint32[:, 1].astype(np.uint32), 8) | \
                    colors_uint32[:, 2].astype(np.uint32)
        rgb_packed_float = rgb_packed.view(np.float32)

        # Combine XYZ and RGB data
        cloud_data = np.column_stack((points, rgb_packed_float))

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # Create header
        header = Header()
        header.frame_id = frame_id
        header.stamp = stamp

        # Create and return PointCloud2 message
        return point_cloud2.create_cloud(header, fields, cloud_data)

    def create_marker_array(self, clusters, bboxes, frame_id):
        """Create a MarkerArray message from the detected clusters."""
        marker_array = MarkerArray()
        for i, cluster in enumerate(clusters):
            if cluster.is_empty():
                continue

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Get bounding box. todo: check if generate_bounding_boxes is true
            # bbox = cluster.get_axis_aligned_bounding_box()
            bbox = bboxes[i]
            if self.bounding_box_type.lower() == "obb":
                center = bbox.center.cpu().numpy().tolist()
                extent = bbox.extent.cpu().numpy().tolist()
                # Convert rotation matrix to quaternion
                R = bbox.rotation.cpu().numpy()
                quat = quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))
                marker.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]),
                                                     z=float(quat[2]), w=float(quat[3]))
            else:
                center = bbox.get_center().cpu().numpy().tolist()
                extent = bbox.get_extent().cpu().numpy().tolist()
                marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            self.get_logger().info(f"center: {center}, type: {type(center)}")
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
            marker_array.markers.append(marker)
        return marker_array

    def create_object_array(self, clusters, bboxes, frame_id):
        """Create an ObjectArray message from the detected clusters."""
        object_array = ObjectArray()
        object_array.header.frame_id = frame_id
        object_array.header.stamp = self.get_clock().now().to_msg()

        for i, cluster in enumerate(clusters):
            if cluster.is_empty():
                continue

            obj = Object()
            obj.detection_level = Object.OBJECT_DETECTED
            # bbox = cluster.get_axis_aligned_bounding_box()
            bbox = bboxes[i]

            if self.bounding_box_type.lower() == "obb":
                center = bbox.center.cpu().numpy().tolist()
                extent = bbox.extent.cpu().numpy().tolist()
                # Convert rotation matrix to quaternion
                R = bbox.rotation.cpu().numpy()
                quat = quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))
                obj.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]),
                                               z=float(quat[2]), w=float(quat[3]))
            else:
                center = bbox.get_center().cpu().numpy().tolist()
                extent = bbox.get_extent().cpu().numpy().tolist()
                obj.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            obj.pose.position.x = center[0]
            obj.pose.position.y = center[1]
            obj.pose.position.z = center[2]

            obj.shape.dimensions = [extent[0], extent[1], extent[2]]

            # Optionally add a class label if classification is implemented
            obj.shape.type = SolidPrimitive().BOX
            obj.id = i
            obj.classification = Object.CLASSIFICATION_UNKNOWN
            obj.classification_certainty = int(125)

            object_array.objects.append(obj)
        return object_array


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
