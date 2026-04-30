#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分割结果可视化工具
专门用于可视化分割标注结果，支持填充颜色显示分割区域

功能特性:
- 只显示分割区域，不显示目标检测框
- 支持分割区域填充颜色
- 支持批量处理和单文件处理
- 实时进度条显示（需要安装tqdm库）
- 详细的处理统计信息
- 支持YOLO格式和JSON格式（包括LabelMe）分割标注
- 支持透明度调节

依赖安装:
pip install tqdm  # 用于更好的进度条显示

使用示例:
    # 批量处理分割标注文件
    python visualize_segmentation.py \
        --image_path /path/to/images \
        --annotation_path /path/to/labels \
        --output_path /path/to/output \
        --batch

    # 单文件处理
    python visualize_segmentation.py \
        --image_path image.jpg \
        --annotation_path annotation.txt \
        --output_path output.jpg

    # 调整透明度
    python visualize_segmentation.py \
        --image_path image.jpg \
        --annotation_path annotation.txt \
        --output_path output.jpg \
        --alpha 0.6

    # 不显示标签
    python visualize_segmentation.py \
        --image_path image.jpg \
        --annotation_path annotation.txt \
        --output_path output.jpg \
        --no_labels
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


class SegmentationVisualizer:
    """分割结果可视化器"""
    
    def __init__(self, class_names: List[str] = None, alpha: float = 0.5, show_labels: bool = True):
        self.class_names = class_names or [
            'pedestrian', 'traffic_cone', 'car', 'pole', 'board', 'box_truck', 'truck_head', 'truck',
            'ground', 'road', 'container_area', 'lock_island', 'AGV', 'smallobs', 'roadblock', 'qc',
            'FL', 'rtgc', 'fence', 'fork_truck', 'van', 'bus', 'goods_vehicle', 'other_vehicle',
            'bicycle', 'bird', 'goods', 'red', 'green', 'yellow', 'grass', 'stone', 'tricycle'
        ]
        
        self.alpha = alpha  # 透明度
        self.show_labels = show_labels  # 是否显示标签
        
        # 为每个类别分配颜色
        self.class_colors = {}
        self.next_color_index = 0
        
        for i, class_name in enumerate(self.class_names):
            self.class_colors[class_name] = COLORS[i % len(COLORS)]
    
    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """为类别获取颜色"""
        if class_name not in self.class_colors:
            self.class_colors[class_name] = COLORS[self.next_color_index % len(COLORS)]
            self.next_color_index += 1
            logger.info(f"为新类别 '{class_name}' 分配颜色: {self.class_colors[class_name]}")
        
        return self.class_colors[class_name]
    
    def parse_yolo_segmentation(self, txt_path: str, img_width: int, img_height: int) -> List[Dict]:
        """解析YOLO格式的分割标注文件"""
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
                        if len(parts) >= 7:  # 分割标注至少需要class_id + 至少3个点
                            class_id = int(parts[0])
                            
                            # 解析分割点坐标
                            points = []
                            for i in range(1, len(parts), 2):
                                if i + 1 < len(parts):
                                    x = float(parts[i]) * img_width
                                    y = float(parts[i + 1]) * img_height
                                    points.append([int(x), int(y)])
                            
                            if len(points) >= 3:  # 至少需要3个点形成多边形
                                if class_id >= len(self.class_names):
                                    logger.warning(f"发现超出范围的类别ID: {class_id}")
                                    if class_id > 1000:
                                        continue
                                
                                annotations.append({
                                    'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}",
                                    'class_id': class_id,
                                    'points': points
                                })
                            else:
                                logger.warning(f"第{line_num}行分割点数量不足: {len(points)} 个点")
                        else:
                            logger.warning(f"第{line_num}行格式错误或不是分割标注: {line}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"第{line_num}行解析失败: {line}, 错误: {e}")
                        
        except Exception as e:
            logger.error(f"读取文件失败 {txt_path}: {e}")
        
        return annotations
    
    def parse_json_segmentation(self, json_path: str) -> List[Dict]:
        """解析JSON格式的分割标注文件（支持LabelMe格式）"""
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
                    
                    # 只处理多边形和矩形分割
                    if shape_type in ['polygon', 'rectangle'] and points:
                        # 转换点坐标
                        converted_points = []
                        for point in points:
                            if len(point) >= 2:
                                converted_points.append([int(point[0]), int(point[1])])
                        
                        if len(converted_points) >= 3:
                            annotations.append({
                                'class_name': label,
                                'class_id': 0,  # LabelMe格式通常没有class_id
                                'points': converted_points,
                                'type': shape_type
                            })
                        elif shape_type == 'rectangle' and len(converted_points) == 2:
                            # 矩形格式：两个对角点
                            x1, y1 = converted_points[0]
                            x2, y2 = converted_points[1]
                            # 转换为四个角点
                            rect_points = [
                                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                            ]
                            annotations.append({
                                'class_name': label,
                                'class_id': 0,
                                'points': rect_points,
                                'type': 'rectangle'
                            })
            
            # 支持COCO格式分割
            elif 'annotations' in data:
                for ann in data['annotations']:
                    segmentation = ann.get('segmentation', [])
                    if segmentation:
                        # COCO格式的分割数据可能是RLE或polygon格式
                        if isinstance(segmentation, list) and len(segmentation) > 0:
                            if isinstance(segmentation[0], list):
                                # polygon格式
                                for seg in segmentation:
                                    if len(seg) >= 6:  # 至少3个点
                                        points = []
                                        for i in range(0, len(seg), 2):
                                            if i + 1 < len(seg):
                                                points.append([int(seg[i]), int(seg[i + 1])])
                                        
                                        if len(points) >= 3:
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
                                                'points': points,
                                                'type': 'polygon'
                                            })
            
            # 支持自定义分割格式
            elif 'segments' in data:
                for segment in data['segments']:
                    points = segment.get('points', [])
                    if len(points) >= 6:  # 至少3个点
                        converted_points = []
                        for i in range(0, len(points), 2):
                            if i + 1 < len(points):
                                converted_points.append([int(points[i]), int(points[i + 1])])
                        
                        if len(converted_points) >= 3:
                            annotations.append({
                                'class_name': segment.get('class', 'unknown'),
                                'class_id': segment.get('class_id', 0),
                                'points': converted_points,
                                'type': 'polygon'
                            })
            
            # 直接包含分割点的格式
            elif 'points' in data:
                points = data['points']
                if len(points) >= 6:  # 至少3个点
                    converted_points = []
                    for i in range(0, len(points), 2):
                        if i + 1 < len(points):
                            converted_points.append([int(points[i]), int(points[i + 1])])
                    
                    if len(converted_points) >= 3:
                        annotations.append({
                            'class_name': data.get('class', 'unknown'),
                            'class_id': data.get('class_id', 0),
                            'points': converted_points,
                            'type': 'polygon'
                        })
        
        except Exception as e:
            logger.error(f"解析JSON文件失败 {json_path}: {e}")
        
        return annotations
    
    def draw_segmentation(self, image: np.ndarray, annotations: List[Dict]) -> np.ndarray:
        """在图像上绘制分割区域（填充颜色）并添加标签"""
        img = image.copy()
        
        # 创建掩码图层用于混合
        overlay = img.copy()
        
        for ann in annotations:
            class_name = ann['class_name']
            points = ann['points']
            
            # 获取颜色
            color = self.get_color_for_class(class_name)
            
            # 创建多边形掩码
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            points_array = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_array], 255)
            
            # 创建彩色掩码
            color_mask = np.zeros_like(img)
            color_mask[mask == 255] = color
            
            # 将彩色掩码叠加到overlay上
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, self.alpha, 0)
        
        # 将原图与overlay混合
        result = cv2.addWeighted(img, 1.0 - self.alpha, overlay, self.alpha, 0)
        
        # 添加类别标签
        if self.show_labels:
            for ann in annotations:
                class_name = ann['class_name']
                points = ann['points']
                color = self.get_color_for_class(class_name)
                
                # 计算标签位置（使用分割区域的中心点）
                if len(points) > 0:
                    # 计算多边形的中心点
                    center_x = sum(p[0] for p in points) // len(points)
                    center_y = sum(p[1] for p in points) // len(points)
                    
                    # 准备标签文本
                    label = f"{class_name}"
                    
                    # 计算文本大小
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # 确保标签在图像范围内
                    label_x = max(10, min(center_x - text_width // 2, img.shape[1] - text_width - 10))
                    label_y = max(text_height + 10, min(center_y, img.shape[0] - 10))
                    
                    # 绘制标签背景
                    cv2.rectangle(result, 
                                 (label_x - 5, label_y - text_height - baseline - 5), 
                                 (label_x + text_width + 5, label_y + baseline + 5), 
                                 color, -1)
                    
                    # 绘制标签边框
                    cv2.rectangle(result, 
                                 (label_x - 5, label_y - text_height - baseline - 5), 
                                 (label_x + text_width + 5, label_y + baseline + 5), 
                                 (255, 255, 255), 1)
                    
                    # 绘制标签文本
                    cv2.putText(result, label, (label_x, label_y), font, font_scale, (255, 255, 255), thickness)
        
        return result
    
    def visualize_single_image(self, image_path: str, annotation_path: str, output_path: str, show_progress: bool = False) -> bool:
        """可视化单张图像的分割结果"""
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
            
            # 解析分割标注文件
            annotations = []
            if annotation_path.lower().endswith('.txt'):
                annotations = self.parse_yolo_segmentation(annotation_path, width, height)
            elif annotation_path.lower().endswith('.json'):
                annotations = self.parse_json_segmentation(annotation_path)
            else:
                logger.error(f"不支持的标注文件格式: {annotation_path}")
                return False
            
            if show_progress:
                print(f"   🏷️  解析到 {len(annotations)} 个分割区域")
            
            # 绘制分割结果
            result_image = self.draw_segmentation(image, annotations)
            
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
            image = cv2.imread(image_path)
            if image is None:
                return
            
            height, width = image.shape[:2]
            
            annotations = []
            if annotation_path.lower().endswith('.txt'):
                annotations = self.parse_yolo_segmentation(annotation_path, width, height)
            elif annotation_path.lower().endswith('.json'):
                annotations = self.parse_json_segmentation(annotation_path)
            
            for ann in annotations:
                class_name = ann['class_name']
                if class_name not in class_stats:
                    class_stats[class_name] = 0
                class_stats[class_name] += 1
                
        except Exception as e:
            logger.warning(f"统计标注信息失败: {e}")
    
    def visualize_batch(self, image_dir: str, annotation_dir: str, output_dir: str) -> dict:
        """批量可视化分割结果"""
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
        class_stats = {}
        
        # 创建进度条
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=len(image_files),
                desc="🎨 分割可视化处理进度",
                unit="张",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
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
    parser = argparse.ArgumentParser(description='分割结果可视化工具')
    parser.add_argument('--image_path', type=str, required=True, 
                       help='图像路径或图像目录')
    parser.add_argument('--annotation_path', type=str, required=True,
                       help='标注文件路径或标注文件目录')
    parser.add_argument('--output_path', type=str, required=True,
                       help='输出路径或输出目录')
    parser.add_argument('--class_names', type=str, nargs='+',
                       help='类别名称列表（可选）')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='分割区域透明度 (0.0-1.0, 默认0.5)')
    parser.add_argument('--no_labels', action='store_true',
                       help='不显示类别标签')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    
    args = parser.parse_args()
    
    # 验证透明度参数
    if not 0.0 <= args.alpha <= 1.0:
        logger.error("透明度参数必须在0.0到1.0之间")
        return
    
    # 创建可视化器
    show_labels = not args.no_labels
    visualizer = SegmentationVisualizer(args.class_names, args.alpha, show_labels)
    
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
        print_summary(result, args.image_path, args.annotation_path, args.output_path, args.alpha, visualizer)
        
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
            logger.info("单文件分割可视化完成")
            print_single_file_summary(args.image_path, args.annotation_path, args.output_path, args.alpha, visualizer)
        else:
            logger.error("单文件分割可视化失败")


def print_summary(result: dict, image_dir: str, annotation_dir: str, output_dir: str, alpha: float, visualizer):
    """打印批量处理的总结信息"""
    print("\n" + "="*60)
    print("🎨 分割可视化处理总结")
    print("="*60)
    
    print(f"📁 输入图像目录: {image_dir}")
    print(f"📁 输入标注目录: {annotation_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎨 透明度设置: {alpha:.2f}")
    print(f"🏷️  标签显示: {'开启' if visualizer.show_labels else '关闭'}")
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
        print(f"🏷️  分割类别统计:")
        total_annotations = sum(result['class_stats'].values())
        print(f"   • 总分割区域数量: {total_annotations}")
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
            print("   📋 正常类别分割区域数量:")
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
        print("✅ 分割可视化处理完成!")
        print("💡 提示: 分割区域已填充颜色，透明度为 {:.2f}".format(alpha))
        if visualizer.show_labels:
            print("💡 提示: 类别标签已显示在每个分割区域中心")
    else:
        print("❌ 分割可视化处理失败!")
    
    print("="*60)


def print_single_file_summary(image_path: str, annotation_path: str, output_path: str, alpha: float, visualizer):
    """打印单文件处理的总结信息"""
    print("\n" + "="*60)
    print("🎨 单文件分割可视化处理总结")
    print("="*60)
    
    print(f"🖼️  输入图像: {image_path}")
    print(f"📄 输入标注: {annotation_path}")
    print(f"💾 输出文件: {output_path}")
    print(f"🎨 透明度设置: {alpha:.2f}")
    print(f"🏷️  标签显示: {'开启' if visualizer.show_labels else '关闭'}")
    print()
    
    # 获取文件大小信息
    try:
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📏 输出文件大小: {file_size_mb:.2f} MB")
    except Exception:
        pass
    
    print("✅ 单文件分割可视化处理完成!")
    print("💡 提示: 分割区域已填充颜色，透明度为 {:.2f}".format(alpha))
    if visualizer.show_labels:
        print("💡 提示: 类别标签已显示在分割区域中心")
    print("="*60)


if __name__ == "__main__":
    main()
