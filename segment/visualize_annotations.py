#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像标注可视化工具
支持JSON和TXT格式的标注文件，包含进度条显示功能

功能特性:
- 支持批量处理和单文件处理
- 实时进度条显示（需要安装tqdm库）
- 详细的处理统计信息
- 支持YOLO格式和JSON格式标注
- 支持分割标注可视化

依赖安装:
pip install tqdm  # 用于更好的进度条显示

使用示例:
    # 批量处理JSON标注文件
    python visualize_annotations.py \
        --image_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/yolov5_dataset/train \
        --annotation_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/yolov5_dataset/predict/exp/json \
        --output_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/yolov5_dataset/predict/exp/json_images_visualize \
        --batch

    # 批量处理TXT标注文件
    python visualize_annotations.py \
        --image_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/yolov5_dataset/train \
        --annotation_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/yolov5_dataset/predict/exp/labels \
        --output_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/yolov5_dataset/predict/exp/label_images_visualize \
        --batch

    # 单文件处理
    python visualize_annotations.py \
        --image_path image.jpg \
        --annotation_path annotation.txt \
        --output_path output.jpg
"""

import cv2
import json
import os
import argparse
import logging
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Union
import numpy as np

# 尝试导入tqdm用于进度条显示
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # 只在直接运行时显示提示，避免在导入时显示
    if __name__ == "__main__":
        print("⚠️  提示: 安装 tqdm 库可以获得更好的进度条显示效果")
        print("   安装命令: pip install tqdm")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预定义颜色列表 - 为33个类别提供足够的颜色
COLORS = [
    (255, 0, 0),      # 红色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 蓝色
    (255, 255, 0),    # 黄色
    (255, 0, 255),    # 紫色
    (0, 255, 255),    # 青色
    (255, 165, 0),    # 橙色
    (128, 0, 128),    # 深紫色
    (0, 128, 0),      # 深绿色
    (128, 128, 0),    # 橄榄色
    (255, 20, 147),   # 深粉色
    (0, 191, 255),    # 深天蓝
    (255, 215, 0),    # 金色
    (138, 43, 226),   # 蓝紫色
    (255, 69, 0),     # 红橙色
    (0, 128, 128),    # 青色
    (255, 105, 180),  # 热粉色
    (34, 139, 34),    # 森林绿
    (255, 140, 0),    # 深橙色
    (75, 0, 130),     # 靛蓝色
    (220, 20, 60),    # 深红色
    (0, 255, 127),    # 春绿色
    (255, 0, 255),    # 洋红色
    (255, 255, 224),  # 浅黄色
    (176, 196, 222),  # 浅钢蓝色
    (255, 182, 193),  # 浅粉色
    (144, 238, 144),  # 浅绿色
    (255, 218, 185),  # 桃色
    (221, 160, 221),  # 梅红色
    (240, 230, 140),  # 卡其色
    (255, 160, 122),  # 浅鲑鱼色
    (230, 230, 250),  # 淡紫色
    (255, 228, 196),  # 莫卡辛色
    (245, 245, 220),  # 米色
    (255, 240, 245),  # 淡紫色
    (240, 248, 255),  # 爱丽丝蓝
    (255, 250, 240),  # 花白色
    (248, 248, 255),  # 幽灵白
    (245, 255, 250),  # 薄荷奶油色
]


class AnnotationVisualizer:
    """标注可视化器"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or [
            'pedestrian', 'traffic_cone', 'car', 'pole', 'board', 'box_truck', 'truck_head', 'truck',
            'ground', 'road', 'container_area', 'lock_island', 'AGV', 'smallobs', 'roadblock', 'qc',
            'FL', 'rtgc', 'fence', 'fork_truck', 'van', 'bus', 'goods_vehicle', 'other_vehicle',
            'bicycle', 'bird', 'goods', 'red', 'green', 'yellow', 'grass', 'stone', 'tricycle'
        ]
        
        # 为每个类别分配颜色
        self.class_colors = {}
        self.next_color_index = 0
        
        # 为预定义的类别分配颜色
        for i, class_name in enumerate(self.class_names):
            self.class_colors[class_name] = COLORS[i % len(COLORS)]
    
    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """为类别获取颜色，如果类别不存在则分配新颜色"""
        if class_name not in self.class_colors:
            # 为新类别分配颜色
            self.class_colors[class_name] = COLORS[self.next_color_index % len(COLORS)]
            self.next_color_index += 1
            logger.info(f"为新类别 '{class_name}' 分配颜色: {self.class_colors[class_name]}")
        
        return self.class_colors[class_name]
    
    def parse_yolo_txt(self, txt_path: str, img_width: int, img_height: int) -> List[Dict]:
        """解析YOLO格式的TXT文件（支持分割标注）"""
        annotations = []
        
        if not os.path.exists(txt_path):
            logger.warning(f"标注文件不存在: {txt_path}")
            return annotations
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            
                            # 检查是否是分割标注（有多个坐标点）
                            if len(parts) > 5:
                                # 分割标注格式：class_id x1 y1 x2 y2 x3 y3 ...
                                points = []
                                for i in range(1, len(parts), 2):
                                    if i + 1 < len(parts):
                                        x = float(parts[i]) * img_width
                                        y = float(parts[i + 1]) * img_height
                                        points.append([int(x), int(y)])
                                
                                if len(points) >= 3:  # 至少需要3个点形成多边形
                                    # 计算边界框
                                    x_coords = [p[0] for p in points]
                                    y_coords = [p[1] for p in points]
                                    x1, x2 = min(x_coords), max(x_coords)
                                    y1, y2 = min(y_coords), max(y_coords)
                                    
                                    # 检查类别ID是否合理
                                    if class_id >= len(self.class_names):
                                        logger.warning(f"发现超出范围的类别ID: {class_id} (最大应为 {len(self.class_names)-1})")
                                        if class_id > 1000:  # 如果ID过大，可能是错误数据
                                            logger.error(f"类别ID {class_id} 过大，可能是标注文件错误，跳过此标注")
                                            continue
                                    
                                    annotations.append({
                                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                                        'class_id': class_id,
                                        'confidence': 1.0,
                                        'bbox': [x1, y1, x2, y2],
                                        'points': points,
                                        'type': 'polygon'
                                    })
                            else:
                                # 标准YOLO格式：class_id x_center y_center width height
                                x_center = float(parts[1]) * img_width
                                y_center = float(parts[2]) * img_height
                                width = float(parts[3]) * img_width
                                height = float(parts[4]) * img_height
                                
                                # 计算边界框坐标
                                x1 = int(x_center - width / 2)
                                y1 = int(y_center - height / 2)
                                x2 = int(x_center + width / 2)
                                y2 = int(y_center + height / 2)
                                
                                # 获取置信度（如果有）
                                confidence = float(parts[5]) if len(parts) > 5 else 1.0
                                
                                # 检查类别ID是否合理
                                if class_id >= len(self.class_names):
                                    logger.warning(f"发现超出范围的类别ID: {class_id} (最大应为 {len(self.class_names)-1})")
                                    if class_id > 1000:  # 如果ID过大，可能是错误数据
                                        logger.error(f"类别ID {class_id} 过大，可能是标注文件错误，跳过此标注")
                                        continue
                                
                                annotations.append({
                                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                                    'class_id': class_id,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2],
                                    'type': 'bbox'
                                })
                        else:
                            logger.warning(f"第{line_num}行格式错误: {line}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"第{line_num}行解析失败: {line}, 错误: {e}")
                        
        except Exception as e:
            logger.error(f"读取文件失败 {txt_path}: {e}")
        
        return annotations
    
    def parse_json_annotations(self, json_path: str) -> List[Dict]:
        """解析JSON格式的标注文件（支持LabelMe格式）"""
        annotations = []
        
        if not os.path.exists(json_path):
            logger.warning(f"标注文件不存在: {json_path}")
            return annotations
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 支持LabelMe格式
            if 'shapes' in data:
                for shape in data['shapes']:
                    label = shape.get('label', 'unknown')
                    shape_type = shape.get('shape_type', 'polygon')
                    points = shape.get('points', [])
                    
                    if points:
                        # 转换点坐标
                        converted_points = []
                        for point in points:
                            if len(point) >= 2:
                                converted_points.append([int(point[0]), int(point[1])])
                        
                        if len(converted_points) >= 3:
                            # 计算边界框
                            x_coords = [p[0] for p in converted_points]
                            y_coords = [p[1] for p in converted_points]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            
                            annotations.append({
                                'class_name': label,
                                'class_id': 0,  # LabelMe格式通常没有class_id
                                'confidence': 1.0,
                                'bbox': [x1, y1, x2, y2],
                                'points': converted_points,
                                'type': shape_type
                            })
            
            # 支持COCO格式
            elif 'annotations' in data:
                for ann in data['annotations']:
                    bbox = ann.get('bbox', [])
                    if len(bbox) == 4:
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h
                        
                        category_id = ann.get('category_id', 0)
                        class_name = f"class_{category_id}"
                        
                        # 尝试从categories中获取类别名称
                        if 'categories' in data:
                            for cat in data['categories']:
                                if cat.get('id') == category_id:
                                    class_name = cat.get('name', class_name)
                                    break
                        
                        annotations.append({
                            'class_name': class_name,
                            'class_id': category_id,
                            'confidence': ann.get('score', 1.0),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'type': 'bbox'
                        })
            
            # 支持自定义格式
            elif 'objects' in data:
                for obj in data['objects']:
                    bbox = obj.get('bbox', [])
                    if len(bbox) == 4:
                        annotations.append({
                            'class_name': obj.get('class', 'unknown'),
                            'class_id': obj.get('class_id', 0),
                            'confidence': obj.get('confidence', 1.0),
                            'bbox': bbox,
                            'type': 'bbox'
                        })
            
            # 直接包含标注的格式
            elif 'bbox' in data:
                bbox = data['bbox']
                if len(bbox) == 4:
                    annotations.append({
                        'class_name': data.get('class', 'unknown'),
                        'class_id': data.get('class_id', 0),
                        'confidence': data.get('confidence', 1.0),
                        'bbox': bbox,
                        'type': 'bbox'
                    })
        
        except Exception as e:
            logger.error(f"解析JSON文件失败 {json_path}: {e}")
        
        return annotations
    
    def draw_annotations(self, image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
        """在图像上绘制标注"""
        img = image.copy()
        
        for ann in annotations:
            class_name = ann['class_name']
            confidence = ann['confidence']
            annotation_type = ann.get('type', 'bbox')
            
            # 获取颜色
            color = self.get_color_for_class(class_name)
            
            if annotation_type == 'polygon' and 'points' in ann:
                # 绘制多边形
                points = np.array(ann['points'], dtype=np.int32)
                cv2.polylines(img, [points], True, color, 2)
                
                # 绘制边界框
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                
            else:
                # 绘制边界框
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # 准备标签文本
            label = f"{class_name}"
            if confidence < 1.0:
                label += f" {confidence:.2f}"
            
            # 计算文本大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 绘制标签背景
            label_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height
            cv2.rectangle(img, (x1, label_y - text_height - baseline), 
                         (x1 + text_width, label_y + baseline), color, -1)
            
            # 绘制标签文本
            cv2.putText(img, label, (x1, label_y), font, font_scale, (255, 255, 255), thickness)
        
        return img
    
    def visualize_single_image(self, image_path: str, annotation_path: str, output_path: str, show_progress: bool = False) -> bool:
        """可视化单张图像"""
        try:
            if show_progress:
                print(f"🔄 正在处理: {os.path.basename(image_path)}")
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"无法读取图像: {image_path}")
                return False
            
            height, width = image.shape[:2]
            if show_progress:
                print(f"   📏 图像尺寸: {width}x{height}")
            
            # 解析标注文件
            annotations = []
            if annotation_path.lower().endswith('.txt'):
                annotations = self.parse_yolo_txt(annotation_path, width, height)
            elif annotation_path.lower().endswith('.json'):
                annotations = self.parse_json_annotations(annotation_path)
            else:
                logger.error(f"不支持的标注文件格式: {annotation_path}")
                return False
            
            if show_progress:
                print(f"   🏷️  解析到 {len(annotations)} 个标注")
            
            # 绘制标注
            result_image = self.draw_annotations(image, annotations)
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, result_image)
            
            if success:
                if show_progress:
                    print(f"   ✅ 结果已保存到: {os.path.basename(output_path)}")
                return True
            else:
                logger.error(f"保存失败: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"处理图像失败: {e}")
            return False
    
    def _update_class_stats(self, image_path: str, annotation_path: str, class_stats: dict):
        """更新类别统计信息"""
        try:
            # 读取图像获取尺寸
            image = cv2.imread(image_path)
            if image is None:
                return
            
            height, width = image.shape[:2]
            
            # 解析标注文件
            annotations = []
            if annotation_path.lower().endswith('.txt'):
                annotations = self.parse_yolo_txt(annotation_path, width, height)
            elif annotation_path.lower().endswith('.json'):
                annotations = self.parse_json_annotations(annotation_path)
            
            # 统计各类别数量
            for ann in annotations:
                class_name = ann['class_name']
                if class_name not in class_stats:
                    class_stats[class_name] = 0
                class_stats[class_name] += 1
                
        except Exception as e:
            logger.warning(f"统计标注信息失败: {e}")
    
    def visualize_batch(self, image_dir: str, annotation_dir: str, output_dir: str) -> dict:
        """批量可视化图像"""
        # 获取所有图像文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
        
        if not image_files:
            logger.error(f"在目录中未找到图像文件: {image_dir}")
            return {'success': False, 'total': 0, 'success_count': 0, 'error_count': 0, 'class_stats': {}}
        
        logger.info(f"找到 {len(image_files)} 张图像")
        
        success_count = 0
        error_count = 0
        class_stats = {}  # 统计各类别的标注数量
        
        # 创建进度条
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=len(image_files),
                desc="🖼️  可视化处理进度",
                unit="张",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            # 简单的进度显示
            print(f"开始处理 {len(image_files)} 张图像...")
            processed_count = 0
        
        for i, image_path in enumerate(image_files):
            try:
                # 构建对应的标注文件路径
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 尝试不同的标注文件格式
                annotation_path = None
                for ext in ['.txt', '.json']:
                    potential_path = os.path.join(annotation_dir, base_name + ext)
                    if os.path.exists(potential_path):
                        annotation_path = potential_path
                        break
                
                if annotation_path is None:
                    logger.warning(f"未找到对应的标注文件: {base_name}")
                    error_count += 1
                    if TQDM_AVAILABLE:
                        pbar.set_postfix_str(f"❌ 缺少标注文件: {base_name}")
                    continue
                
                # 构建输出路径
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                
                # 可视化
                if self.visualize_single_image(image_path, annotation_path, output_path):
                    success_count += 1
                    # 统计标注信息
                    self._update_class_stats(image_path, annotation_path, class_stats)
                    if TQDM_AVAILABLE:
                        pbar.set_postfix_str(f"✅ {os.path.basename(image_path)}")
                else:
                    error_count += 1
                    if TQDM_AVAILABLE:
                        pbar.set_postfix_str(f"❌ 处理失败: {os.path.basename(image_path)}")
                    
            except Exception as e:
                logger.error(f"处理图像失败 {image_path}: {e}")
                error_count += 1
                if TQDM_AVAILABLE:
                    pbar.set_postfix_str(f"❌ 异常: {os.path.basename(image_path)}")
            
            # 更新进度条
            if TQDM_AVAILABLE:
                pbar.update(1)
            else:
                processed_count += 1
                if processed_count % 10 == 0 or processed_count == len(image_files):
                    progress = processed_count / len(image_files) * 100
                    print(f"进度: {processed_count}/{len(image_files)} ({progress:.1f}%) - 成功: {success_count}, 失败: {error_count}")
        
        # 关闭进度条
        if TQDM_AVAILABLE:
            pbar.close()
        else:
            print(f"处理完成! 成功: {success_count}, 失败: {error_count}")
        
        result = {
            'success': success_count > 0,
            'total': len(image_files),
            'success_count': success_count,
            'error_count': error_count,
            'class_stats': class_stats
        }
        
        return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像标注可视化工具')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='图像路径或图像目录')
    parser.add_argument('--annotation_path', type=str, required=True,
                       help='标注文件路径或标注文件目录')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出路径或输出目录')
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='类别名称列表（可选）')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = AnnotationVisualizer(args.class_names)
    
    if args.batch:
        # 批量处理模式
        if not os.path.isdir(args.image_path):
            logger.error("批量模式需要指定图像目录")
            return
        
        if not os.path.isdir(args.annotation_path):
            logger.error("批量模式需要指定标注文件目录")
            return
        
        os.makedirs(args.output_path, exist_ok=True)
        result = visualizer.visualize_batch(args.image_path, args.annotation_path, args.output_path)
        
        # 输出详细的总结信息
        print_summary(result, args.image_path, args.annotation_path, args.output_path)
        
    else:
        # 单文件处理模式
        if not os.path.isfile(args.image_path):
            logger.error("单文件模式需要指定图像文件")
            return
        
        if not os.path.isfile(args.annotation_path):
            logger.error("单文件模式需要指定标注文件")
            return
        
        success = visualizer.visualize_single_image(args.image_path, args.annotation_path, args.output_path, show_progress=True)
        
        if success:
            logger.info("单文件可视化完成")
            print_single_file_summary(args.image_path, args.annotation_path, args.output_path)
        else:
            logger.error("单文件可视化失败")


def print_summary(result: dict, image_dir: str, annotation_dir: str, output_dir: str):
    """打印批量处理的总结信息"""
    print("\n" + "="*60)
    print("📊 批量可视化处理总结")
    print("="*60)
    
    print(f"📁 输入图像目录: {image_dir}")
    print(f"📁 输入标注目录: {annotation_dir}")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    print(f"📈 处理统计:")
    print(f"   • 总图像数量: {result['total']}")
    print(f"   • 成功处理: {result['success_count']} 张")
    print(f"   • 处理失败: {result['error_count']} 张")
    success_rate = result['success_count']/result['total']*100 if result['total'] > 0 else 0
    print(f"   • 成功率: {success_rate:.1f}%")
    
    # 添加进度条样式的成功率显示
    if result['total'] > 0:
        bar_length = 20
        filled_length = int(bar_length * result['success_count'] / result['total'])
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"   • 进度条: [{bar}] {success_rate:.1f}%")
    print()
    
    if result['class_stats']:
        print(f"🏷️  类别统计:")
        total_annotations = sum(result['class_stats'].values())
        print(f"   • 总标注数量: {total_annotations}")
        print(f"   • 类别数量: {len(result['class_stats'])}")
        print()
        
        # 分离正常类别和异常类别
        normal_classes = {}
        abnormal_classes = {}
        for class_name, count in result['class_stats'].items():
            if class_name.startswith('class_') and class_name != 'class_':
                abnormal_classes[class_name] = count
            else:
                normal_classes[class_name] = count
        
        # 按数量排序显示正常类别
        if normal_classes:
            sorted_classes = sorted(normal_classes.items(), key=lambda x: x[1], reverse=True)
            print("   📋 正常类别标注数量:")
            for class_name, count in sorted_classes:
                percentage = count / total_annotations * 100 if total_annotations > 0 else 0
                print(f"      • {class_name}: {count} ({percentage:.1f}%)")
            print()
        
        # 显示异常类别警告
        if abnormal_classes:
            print("   ⚠️  发现异常类别ID:")
            for class_name, count in abnormal_classes.items():
                percentage = count / total_annotations * 100 if total_annotations > 0 else 0
                print(f"      • {class_name}: {count} ({percentage:.1f}%) - 请检查标注文件")
            print()
            print("   💡 建议: 检查标注文件中的类别ID是否正确")
            print()
    
    if result['success']:
        print("✅ 批量可视化处理完成!")
    else:
        print("❌ 批量可视化处理失败!")
    
    print("="*60)


def print_single_file_summary(image_path: str, annotation_path: str, output_path: str):
    """打印单文件处理的总结信息"""
    print("\n" + "="*60)
    print("📊 单文件可视化处理总结")
    print("="*60)
    
    print(f"🖼️  输入图像: {image_path}")
    print(f"📄 输入标注: {annotation_path}")
    print(f"💾 输出文件: {output_path}")
    print()
    
    # 获取文件大小信息
    try:
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📏 输出文件大小: {file_size_mb:.2f} MB")
    except Exception:
        pass
    
    print("✅ 单文件可视化处理完成!")
    print("="*60)


if __name__ == "__main__":
    main()