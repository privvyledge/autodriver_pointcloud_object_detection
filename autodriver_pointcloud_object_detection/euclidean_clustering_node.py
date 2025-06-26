"""
ROS2 node for obstacle detection using Euclidean Clustering on Point Clouds.
Publishes clustered pointcloud, visualization markers, and object array.

Usage:
    sudo apt-get install ros-${ROS_DISTRO}-derived-object-msgs
    ros2 run pointcloud_obstacle_detection euclidean_clustering_node

Todo:
    * publish cluster labels
    * publish labels colormap
    * pass namespace to preprocessing params
    * switch to self.get_parameter_or(name, alternative_value=None) instead of self.get_parameter(name)
    * project the pointcloud to an  image frame
        i. Copy the current image and camera info in the pointcloud callback (or use message_filters)
        ii. (optional) project the pointcloud to the robots frame
        iii. (optional) preprocess the pointcloud
        iv. Detect clusters and get 3D bounding boxes
        v. Publish markers, detection object array, and combined cluster pointcloud
        vi. (optional) if self.project_clusters_to_image
            a. Get the transform from the output frame (either robot or pointcloud frame) to the image frame
            b. Project the combined cluster pointcloud (or each cluster separately) and bounding boxes to the image frame
            c. (optional) draw a 2D bounding box around each cluster using the min and max x and y coordinates
            d. (optional) if classifying: send the bounding boxes as batches to the classifier, e.g dinov2
            e. publish the image, bounding boxes and optionally labels
            f. (optional) add the image detection labels to the pointcloud and republish the pointcloud detected objects
    * tune parameters for speed and accuracy
    * Upsample the detected cluster using the bounding box extent then recluster based on the original point cloud
    * Use python composition to run this node with pointcloud preprocessor
"""
import sys
import time
import struct
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

from autodriver_pointcloud_preprocessor.pointcloud_preprocessor import PointcloudPreprocessorNode
from autodriver_pointcloud_preprocessor.utils import (convert_pointcloud_to_numpy, numpy_struct_to_pointcloud2,
                                                      get_current_time, get_time_difference,
                                                      dict_to_open3d_tensor_pointcloud,
                                                      pointcloud_to_dict, get_pointcloud_metadata,
                                                      check_field, crop_pointcloud,
                                                      extract_rgb_from_pointcloud, get_fields_from_dicts,
                                                      remove_duplicates, rgb_int_to_float,
                                                      FIELD_DTYPE_MAP, FIELD_DTYPE_MAP_INV)


class ObstacleDetectionNode(PointcloudPreprocessorNode):
    def __init__(self):
        # See PointcloudPreprocessorNode for parameters and variables initialized
        super(ObstacleDetectionNode, self).__init__("obstacle_detection_node", enabled=False,
                                                    parameter_namespace='pointcloud_preprocessor')

        # Declare parameters
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
        self.declare_parameter('markers_topic', '/obstacle_markers')
        self.declare_parameter('objects_topic', '/detected_objects')
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

        self.declare_parameter("cluster_tolerance", 0.2)  # meters
        self.declare_parameter("min_cluster_size", 100)
        self.declare_parameter("max_cluster_size", 10000)
        self.declare_parameter('cluster_min_height', 0.1)  # min height of cluster
        self.declare_parameter('cluster_max_height', 2.0)  # max height of cluster

        self.declare_parameter("generate_bounding_box", True)
        self.declare_parameter("bounding_box_type", "AABB")  # AABB or OBB

        # self.declare_parameter("generate_convex_hull", False)

        self.declare_parameter("project_clusters_to_image", False)
        self.declare_parameter("classify_image_clusters", False)
        self.declare_parameter(name='input_image_topic', value="/camera/camera/color/image_raw",
                               descriptor=ParameterDescriptor(
                                       description='The input image topic. '
                                                   'Works with all image types: RGB(A), BGR(A), mono8, mono16.',
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
        self.cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        self.max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value
        self.cluster_min_height = self.get_parameter('cluster_min_height').value
        self.cluster_max_height = self.get_parameter('cluster_max_height').value
        self.generate_bounding_box = self.get_parameter("generate_bounding_box").value
        self.bounding_box_type = self.get_parameter("bounding_box_type").value
        # self.generate_convex_hull = self.get_parameter("generate_convex_hull").value

        self.project_clusters_to_image = self.get_parameter("project_clusters_to_image").value
        self.classify_image_clusters = self.get_parameter("classify_image_clusters").value
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

        # Setup subscribers
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
            self.extract_pointcloud(ros_cloud)

            # preprocess the PointCloud
            preprocessing_start_time = get_current_time(monotonic=False)
            self.preprocess()
            self.processing_times['preprocessing_time'] = get_time_difference(preprocessing_start_time,
                                                                              get_current_time(monotonic=False))
            # Detect obstacles
            start_time = get_current_time(monotonic=False)
            # Detect obstacles and get clusters
            clusters, bboxes = self.detect_obstacles()
            self.processing_times['detect_obstacles'] = get_time_difference(start_time,
                                                                            get_current_time(monotonic=False))
            # self.get_logger().info(f"Cluster {clusters}, length: {len(clusters)}, bbox length: {len(bboxes)}")

            if clusters:
                # Publish clustered pointcloud
                start_time = get_current_time(monotonic=True)
                clustered_cloud = self.combine_clusters(clusters)
                self.processing_times['combine_clusters'] = get_time_difference(start_time, get_current_time(monotonic=True))

                if clustered_cloud is not None:
                    # Publish processed point cloud
                    start_time = get_current_time(monotonic=True)
                    processed_struct = self.prepare_pointcloud(ros_cloud)
                    new_header = self.create_header(ros_cloud)

                    pc_msg = self.tensor_to_ros_cloud(processed_struct, self.pointfields, header=new_header)
                    pc_msg.is_dense = self.remove_nans and self.remove_infs
                    start_time = get_current_time(monotonic=True)
                    self.pointcloud_pub.publish(pc_msg)
                    self.processing_times['pointcloud_pub'] = get_time_difference(start_time,
                                                                                  get_current_time(monotonic=True))

                    pcd_number = str(self.frame_count).zfill(8)
                    self.pointcloud_saver(pcd_number)
                    self.pointcloud_visualizer(pcd_number)

                    self.frame_count += 1

                # Create and publish markers
                frame_id = self.robot_frame if self.camera_to_robot_tf else ros_cloud.header.frame_id  # if self.robot_frame
                marker_array = self.create_marker_array(clusters, bboxes, frame_id=frame_id)
                self.marker_pub.publish(marker_array)

                # Create and publish object array
                object_array = self.create_object_array(clusters, bboxes, frame_id=frame_id)
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
            self.get_logger().error(f"Error processing point cloud: {str(e)}")

    def detect_obstacles(self):
        """
        Perform DBSCAN clustering and return clusters with their bounding boxes.
        Return the list of clusters.
        Todo: remove clusters with more points than self.max_cluster_size
        """
        # self.get_logger().info("Clustering point cloud with DBSCAN...")

        # Ensure the point cloud is in memory
        if self.o3d_pointcloud.is_empty():
            # self.get_logger().info(f"Pointcloud is empty")
            return [], []

        # Use GPU DBSCAN clustering.
        labels = self.o3d_pointcloud.cluster_dbscan(
                eps=self.cluster_tolerance,
                min_points=self.min_cluster_size,
                print_progress=True  # todo: set to False
        )
        self.o3d_pointcloud.point.labels = labels  # if adding label, create a new Pointcloud Tensor Geometry object in each callback or clear

        # Get unique labels (excluding noise points labeled as -1)
        labels = torch.utils.dlpack.from_dlpack(labels.to_dlpack())
        # labels = labels.cpu().numpy()
        # unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)  # largest_cluster_label = max(labels, key=lambda l: np.sum(labels == l))
        unique_labels, counts = torch.unique(labels[labels != -1], return_counts=True)  # todo: switch to unique_consecutive

        if len(unique_labels) == 0:
            # self.get_logger().info(f"unique labels: {unique_labels}, len: {len(unique_labels)}")
            return [], []  # Return empty lists if no valid clusters

        #unique_labels = unique_labels[unique_labels >= 0]

        max_label = labels.max().item()
        # self.get_logger().info(f"DBSCAN found {max_label + 1} clusters")

        clusters = []
        bboxes = []

        for label in unique_labels:
            # Create mask for current cluster
            mask = (labels == label)

            # todo: remove clusters with more points than self.max_cluster_size, use counts or mask length

            # we convert the boolean array to an integer array since dlpack does not support zero-copy transfer for bool
            mask = mask.to(device=self.torch_device, dtype=torch.uint8)
            mask = o3c.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(mask))  # o3c.Tensor(mask, device=self.o3d_device)
            mask = mask.to(o3c.Dtype.Bool)  # convert back to a boolean mask

            # Create new pointcloud for cluster
            cluster_pcd = self.o3d_pointcloud.select_by_mask(mask)

            # # Get cluster height. todo: use mask to remove clusters with min_height > threshold
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

    def create_marker_array(self, clusters, bboxes, frame_id):
        """Create a MarkerArray message from the detected clusters.
        Todo: speed up by looping through clusters once instead of for each message"""
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

            # self.get_logger().info(f"center: {center}, type: {type(center)}")
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
