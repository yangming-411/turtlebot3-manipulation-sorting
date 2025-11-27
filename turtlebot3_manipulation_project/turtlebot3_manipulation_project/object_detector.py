#!/usr/bin/env python3
"""
object_detector.py - 基础物体识别器
第一步：颜色识别
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        
        # 创建CV桥接器
        self.bridge = CvBridge()
        
        # 订阅相机图像
        self.image_sub = self.create_subscription(
            Image,
            '/pi_camera/image_raw',
            self.image_callback,
            10
        )
        
        # 定义颜色范围 (HSV格式)
        self.color_ranges = {
            'red': [
                ([0, 120, 70], [10, 255, 255]),      # 红色范围1
                ([170, 120, 70], [180, 255, 255])    # 红色范围2
            ],
            'blue': [
                ([100, 150, 50], [130, 255, 255])    # 蓝色范围
            ],
            'yellow': [
                ([20, 100, 100], [30, 255, 255])     # 黄色范围
            ]
        }
        
        self.get_logger().info("物体识别器初始化完成，等待图像数据...")
    
    def image_callback(self, msg):
        try:
            # 转换ROS图像为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 转换为HSV颜色空间
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 识别各种颜色的物体
            detections = self.detect_colors(hsv_image, cv_image)
            
            # 显示结果
            self.display_results(cv_image, detections)
            
        except Exception as e:
            self.get_logger().error(f"图像处理错误: {str(e)}")
    
    def detect_colors(self, hsv_image, original_image):
        """识别不同颜色的物体"""
        detections = []
        
        for color_name, ranges in self.color_ranges.items():
            # 创建颜色掩码
            mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
            
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                color_mask = cv2.inRange(hsv_image, lower, upper)
                mask = cv2.bitwise_or(mask, color_mask)
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 150:  # 过滤小面积噪声
                    # 计算边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    detections.append({
                        'color': color_name,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w//2, y + h//2)
                    })
        
        return detections
    
    def display_results(self, image, detections):
        """在图像上显示识别结果"""
        display_image = image.copy()
        
        for detection in detections:
            color = detection['color']
            x, y, w, h = detection['bbox']
            center_x, center_y = detection['center']
            
            # 根据颜色设置绘制颜色
            if color == 'red':
                bgr_color = (0, 0, 255)  # 红色
            elif color == 'blue':
                bgr_color = (255, 0, 0)  # 蓝色
            else:  # yellow
                bgr_color = (0, 255, 255)  # 黄色
            
            # 绘制边界框和中心点
            cv2.rectangle(display_image, (x, y), (x + w, y + h), bgr_color, 2)
            cv2.circle(display_image, (center_x, center_y), 5, bgr_color, -1)
            
            # 添加标签
            label = f"{color} ({detection['area']:.0f})"
            cv2.putText(display_image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)
        
        # 显示图像
        cv2.imshow('Object Detection - Color Only', display_image)
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