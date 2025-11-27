#!/usr/bin/env python3
"""
object_detector.py - 完整物体识别器
第三步：颜色 + 形状 + 位置信息
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
import cv2
import numpy as np
from cv_bridge import CvBridge

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        self.bridge = CvBridge()
        
        # 订阅RGB图像和深度图像
        self.image_sub = self.create_subscription(
            Image, '/pi_camera/image_raw', self.image_callback, 10)
        
        self.depth_sub = self.create_subscription(
            Image, '/pi_camera/depth/image_raw', self.depth_callback, 10)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/pi_camera/camera_info', self.camera_info_callback, 10)
        
        # 发布检测到的物体位置
        self.detection_pub = self.create_publisher(PointStamped, '/detected_objects', 10)
        
        # 存储深度图像和相机参数
        self.depth_image = None
        self.camera_info = None
        
        # 颜色和形状定义
        self.color_ranges = {
            'red': [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])],
            'blue': [([100, 150, 50], [130, 255, 255])],
            'yellow': [([20, 100, 100], [30, 255, 255])]
        }
        
        self.expected_shapes = {
            'red': 'cube',
            'blue': 'cylinder', 
            'yellow': 'sphere'
        }
        
        self.get_logger().info("完整物体识别器初始化完成")
    
    def camera_info_callback(self, msg):
        """存储相机参数"""
        self.camera_info = msg
        self.get_logger().info("收到相机参数")
    
    def depth_callback(self, msg):
        """存储深度图像"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            self.get_logger().error(f"深度图像转换错误: {str(e)}")
    
    def pixel_to_3d(self, u, v, depth):
        """将像素坐标转换为3D坐标"""
        if self.camera_info is None or depth <= 0:
            return None
        
        fx = self.camera_info.k[0]  # 焦距 x
        fy = self.camera_info.k[4]  # 焦距 y
        cx = self.camera_info.k[2]  # 主点 x
        cy = self.camera_info.k[5]  # 主点 y
        
        # 计算3D坐标
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return (x, y, z)
    
    def detect_shape(self, contour):
        """改进的形状识别"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return "unknown", 0.0
        
        # 多种形状因子
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # 多边形近似
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        vertices = len(approx)
        
        # 计算边界框特征
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # 形状判断
        if circularity > 0.85:
            return "sphere", circularity
        elif vertices == 4 and 0.7 <= aspect_ratio <= 1.3:
            return "cube", 0.8
        elif circularity > 0.65:
            return "cylinder", circularity
        else:
            return "unknown", 0.0
    
    def image_callback(self, msg):
        if self.depth_image is None:
            self.get_logger().warn("等待深度图像...")
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            detections = self.detect_objects(hsv_image, cv_image)
            self.display_results(cv_image, detections)
            self.publish_detections(detections)
            
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")
    
    def detect_objects(self, hsv_image, original_image):
        """完整的物体检测"""
        detections = []
        
        for color_name, ranges in self.color_ranges.items():
            expected_shape = self.expected_shapes[color_name]
            
            # 颜色分割
            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                color_mask = cv2.inRange(hsv_image, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # 噪声去除
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:
                    continue
                
                # 形状识别
                shape, shape_confidence = self.detect_shape(contour)
                
                # 几何特征
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w//2, y + h//2
                
                # 获取深度信息
                depth = self.get_depth_value(center_x, center_y)
                position_3d = self.pixel_to_3d(center_x, center_y, depth) if depth else None
                
                # 计算置信度
                confidence = self.calculate_confidence(area, shape, expected_shape, shape_confidence)
                
                detection = {
                    'color': color_name,
                    'shape': shape,
                    'expected_shape': expected_shape,
                    'shape_match': shape == expected_shape,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'center_2d': (center_x, center_y),
                    'depth': depth,
                    'position_3d': position_3d,
                    'confidence': confidence
                }
                
                detections.append(detection)
        
        return detections
    
    def get_depth_value(self, x, y):
        """获取深度值（使用小区域平均）"""
        if self.depth_image is None:
            return None
        
        x, y = int(x), int(y)
        h, w = self.depth_image.shape
        
        # 检查边界
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
        
        # 取小区域平均深度
        depth_roi = self.depth_image[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
        valid_depths = depth_roi[depth_roi > 0]
        
        if len(valid_depths) == 0:
            return None
        
        return float(np.mean(valid_depths))
    
    def calculate_confidence(self, area, detected_shape, expected_shape, shape_confidence):
        """计算综合置信度"""
        confidence = 0.3  # 基础置信度
        
        # 面积置信度 (0-0.3)
        area_conf = min(area / 3000, 0.3)
        confidence += area_conf
        
        # 形状匹配置信度
        if detected_shape == expected_shape:
            confidence += 0.3
            confidence += shape_confidence * 0.1
        
        return min(confidence, 1.0)
    
    def publish_detections(self, detections):
        """发布检测结果"""
        for detection in detections:
            if detection['position_3d'] and detection['confidence'] > 0.6:
                point_msg = PointStamped()
                point_msg.header.stamp = self.get_clock().now().to_msg()
                point_msg.header.frame_id = "camera_rgb_optical_frame"
                
                x, y, z = detection['position_3d']
                point_msg.point.x = x
                point_msg.point.y = y
                point_msg.point.z = z
                
                self.detection_pub.publish(point_msg)
    
    def display_results(self, image, detections):
        """显示详细结果"""
        display_image = image.copy()
        
        for detection in detections:
            color_name = detection['color']
            shape = detection['shape']
            match = detection['shape_match']
            confidence = detection['confidence']
            position_3d = detection['position_3d']
            
            # 颜色设置
            if color_name == 'red':
                bgr_color = (0, 0, 255)
            elif color_name == 'blue':
                bgr_color = (255, 0, 0)
            else:
                bgr_color = (0, 255, 255)
            
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center_2d']
            
            # 绘制
            box_color = (0, 255, 0) if match else (0, 0, 255)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), box_color, 2)
            cv2.circle(display_image, (center_x, center_y), 5, bgr_color, -1)
            
            # 标签信息
            status = "✓" if match else "✗"
            labels = [
                f"{color_name} {shape} {status}",
                f"Conf: {confidence:.2f}",
                f"Area: {detection['area']:.0f}"
            ]
            
            if position_3d:
                labels.append(f"Pos: ({position_3d[0]:.2f}, {position_3d[1]:.2f}, {position_3d[2]:.2f})")
            
            for i, label in enumerate(labels):
                cv2.putText(display_image, label, (x, y - 5 - i*15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_color, 1)
        
        # 统计信息
        total = len(detections)
        valid = sum(1 for d in detections if d['confidence'] > 0.6)
        stats = f"Total: {total} | Valid: {valid}"
        cv2.putText(display_image, stats, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Object Detection - Complete', display_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    detector = ObjectDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        detector.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()