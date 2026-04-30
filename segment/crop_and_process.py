#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像裁剪和标注文件处理工具

功能说明：
1. 根据给定的裁剪参数对原图像进行裁剪
2. 将裁剪后的图像缩放到指定尺寸（默认是原图像大小）并保存
3. 处理labelme格式的标注文件, 将点坐标转换为mask, 然后裁剪并缩放
4. 只保留在裁剪区域内的mask, 区域外的mask不保存
5. 更新imageData字段为裁剪后缩放图像的base64编码, 确保labelme正确显示新图像
6. 保存裁剪后缩放的图像和新的标注文件

使用方法：
    python segment/crop_and_process.py \
        --image_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/dataset_250827/images \
        --json_path /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/dataset_250827/labelme/total2017 \
        --videocrop left=100 top=200 right=380 bottom=298 \
        --target_size width=960 height=768 \
        --suffix _center \
        --output_image_dir /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/dataset_250828_center/images \
        --output_json_dir /home/hzh/qpilot3_web/zhhu/yolov5_data_model/data/dataset_250828_center/images

参数说明：
    --image_path: 原图像文件夹路径
    --json_path: labelme标注文件文件夹路径
    --videocrop: 裁剪参数，格式为 left=100 top=200 right=380 bottom=298
    --target_size: 目标图像尺寸, 格式为 width=640 height=480。如果不提供, 默认使用原图像尺寸
    --suffix: 裁剪后图像的文件名后缀，默认为 _cropped。可以设置为空字符串 "" 以保持原文件名
    --output_image_dir: 裁剪后图像保存目录，默认为 ./cropped_images
    --output_json_dir: 裁剪后标注文件保存目录，默认为 ./cropped_jsons
"""

import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
import shutil
from tqdm import tqdm
import base64


def parse_crop_params(crop_str: str) -> Dict[str, int]:
    """
    解析裁剪参数字符串
    
    Args:
        crop_str: 裁剪参数字符串，格式为 "left=100 top=200 right=380 bottom=298"
    
    Returns:
        包含裁剪参数的字典
    """
    params = {}
    parts = crop_str.split()
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            params[key] = int(value)
    
    return params


def parse_target_size(size_str: str) -> Tuple[int, int]:
    """
    解析目标尺寸字符串
    
    Args:
        size_str: 目标尺寸字符串，格式为 "width=640 height=480"
    
    Returns:
        目标尺寸元组 (width, height)
    """
    params = {}
    parts = size_str.split()
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            params[key] = int(value)
    
    width = params.get('width', None)
    height = params.get('height', None)
    
    if width is None or height is None:
        raise ValueError(f"目标尺寸格式错误，需要 width=xxx height=xxx, 实际输入: {size_str}")
    
    return width, height


def calculate_crop_coordinates(crop_params: Dict[str, int], width: int, height: int, is_explicit: bool = False) -> Tuple[int, int, int, int]:
    """
    计算裁剪坐标
    
    Args:
        crop_params: 裁剪参数字典
        width: 图像宽度
        height: 图像高度
        is_explicit: 是否为用户明确指定的参数(True表示用户明确指定, False表示使用默认值)
    
    Returns:
        裁剪坐标 (left, top, right, bottom)
    """
    # 获取裁剪参数（距离边界的像素数）
    left_dist = crop_params.get('left', 0)      # 距离左边的像素数
    top_dist = crop_params.get('top', 0)        # 距离顶部的像素数
    right_dist = crop_params.get('right', 0)    # 距离右边的像素数
    bottom_dist = crop_params.get('bottom', 0)  # 距离底部的像素数
    
    # 如果所有参数都是0
    if left_dist == 0 and top_dist == 0 and right_dist == 0 and bottom_dist == 0:
        if is_explicit:
            # 用户明确指定全0，表示不裁剪，返回整个图像
            return 0, 0, width, height
        else:
            # 默认参数，裁剪左上角1/4区域
            crop_left = 0
            crop_top = 0
            crop_right = width // 2
            crop_bottom = height // 2
    else:
        # 计算实际的裁剪坐标
        crop_left = left_dist
        crop_top = top_dist
        crop_right = width - right_dist
        crop_bottom = height - bottom_dist
    
    # 确保裁剪区域有效
    crop_left = max(0, min(crop_left, width - 1))
    crop_top = max(0, min(crop_top, height - 1))
    crop_right = max(crop_left + 1, min(crop_right, width))
    crop_bottom = max(crop_top + 1, min(crop_bottom, height))
    
    return crop_left, crop_top, crop_right, crop_bottom


def points_to_mask(points: List[List[float]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    将点坐标列表转换为mask
    
    Args:
        points: 点坐标列表，格式为 [[x1, y1], [x2, y2], ...]
        image_shape: 图像形状 (height, width)
    
    Returns:
        二值化mask数组
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if len(points) < 3:
        return mask
    
    # 将点坐标转换为整数
    points_array = np.array(points, dtype=np.int32)
    
    # 填充多边形区域
    cv2.fillPoly(mask, [points_array], 255)
    
    return mask


def mask_to_points(mask: np.ndarray) -> List[List[float]]:
    """
    将mask转换回点坐标列表
    
    Args:
        mask: 二值化mask数组
    
    Returns:
        点坐标列表
    """
    # 找到轮廓，使用CHAIN_APPROX_NONE获取所有点，不进行简化
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return []
    
    # 选择最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 获取图像尺寸
    height, width = mask.shape[:2]
    
    # 转换为点坐标列表，确保坐标在有效范围内
    points = []
    for point in largest_contour:
        x, y = float(point[0][0]), float(point[0][1])
        # 确保坐标在图像范围内
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        points.append([x, y])
    
    # 确保至少有3个点
    if len(points) < 3:
        return []
    
    return points


def crop_and_transform_mask(mask: np.ndarray, crop_params: Dict[str, int], 
                           original_shape: Tuple[int, int], target_size: Tuple[int, int] = None, is_explicit: bool = False) -> Tuple[np.ndarray, bool, bool]:
    """
    裁剪并变换mask, 适应裁剪后缩放到目标尺寸的图像
    
    Args:
        mask: 原始mask
        crop_params: 裁剪参数（距离边界的像素数）
        original_shape: 原始图像形状
        target_size: 目标图像尺寸 (width, height), 如果为None则使用原图尺寸
        is_explicit: 是否为用户明确指定的参数
    
    Returns:
        变换后的mask、是否保留完整mask的标志、是否在裁剪区域内
    """
    height, width = original_shape[:2]
    
    # 计算裁剪坐标
    crop_left, crop_top, crop_right, crop_bottom = calculate_crop_coordinates(crop_params, width, height, is_explicit)
    
    # 裁剪mask
    cropped_mask = mask[crop_top:crop_bottom, crop_left:crop_right]
    
    # 检查裁剪后的mask是否还有有效区域
    if not np.any(cropped_mask > 0):
        # mask完全在裁剪区域外，不保存
        return None, False, False
    
    # 确定目标尺寸
    if target_size is None:
        target_width, target_height = width, height
    else:
        target_width, target_height = target_size
    
    # 将裁剪后的mask缩放到目标尺寸，使用最近邻插值保持mask的清晰边界
    resized_mask = cv2.resize(cropped_mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    return resized_mask, False, True


def process_labelme_json(json_path: str, crop_params: Dict[str, int], 
                        cropped_shape: Tuple[int, int], new_image_filename: str, original_shape: Tuple[int, int], cropped_image: np.ndarray, target_size: Tuple[int, int] = None, is_explicit: bool = False) -> Dict[str, Any]:
    """
    处理labelme标注文件
    
    Args:
        json_path: labelme标注文件路径
        crop_params: 裁剪参数
        cropped_shape: 裁剪后的图像形状
        new_image_filename: 新的图像文件名
        original_shape: 原始图像形状
        cropped_image: 裁剪后的图像数据
        target_size: 目标图像尺寸 (width, height), 如果为None则使用原图尺寸
        is_explicit: 是否为用户明确指定的参数
    
    Returns:
        处理后的标注数据
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保必要的字段存在
    if 'version' not in data:
        data['version'] = '5.0.1'
    if 'flags' not in data:
        data['flags'] = {}
    if 'shapes' not in data:
        data['shapes'] = []
    
    # 更新图像路径和尺寸
    data['imagePath'] = new_image_filename
    data['imageWidth'] = cropped_shape[1]
    data['imageHeight'] = cropped_shape[0]
    
    # 将裁剪后的图像编码为base64并更新imageData字段
    _, buffer = cv2.imencode('.jpg', cropped_image)
    image_data = base64.b64encode(buffer).decode('utf-8')
    data['imageData'] = image_data
    
    # 获取裁剪坐标
    height, width = original_shape[:2]
    crop_left, crop_top, crop_right, crop_bottom = calculate_crop_coordinates(crop_params, width, height, is_explicit)
    
    # 处理每个标注
    new_shapes = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            # 将点坐标转换为mask
            points = shape['points']
            mask = points_to_mask(points, (height, width))
            
            # 裁剪并变换mask（适应缩放后的图像）
            transformed_mask, keep_full, in_crop_area = crop_and_transform_mask(mask, crop_params, (height, width), target_size, is_explicit)
            
            # 只保存在裁剪区域内的mask
            if in_crop_area and transformed_mask is not None:
                new_points = mask_to_points(transformed_mask)
                if new_points and len(new_points) >= 3:  # 至少需要3个点形成多边形
                    # 额外验证：确保所有点都在图像范围内
                    valid_points = []
                    for point in new_points:
                        x, y = point[0], point[1]
                        if 0 <= x < cropped_shape[1] and 0 <= y < cropped_shape[0]:
                            valid_points.append(point)
                    
                    # 确保验证后仍有足够的点
                    if len(valid_points) >= 3:
                        shape['points'] = valid_points
                        new_shapes.append(shape)
    
    data['shapes'] = new_shapes
    return data


def crop_image(image_path: str, crop_params: Dict[str, int], suffix: str, target_size: Tuple[int, int] = None, is_explicit: bool = False) -> Tuple[np.ndarray, str]:
    """
    裁剪图像并缩放到目标尺寸
    
    Args:
        image_path: 图像文件路径
        crop_params: 裁剪参数（距离边界的像素数）
        suffix: 文件名后缀
        target_size: 目标图像尺寸 (width, height), 如果为None则使用原图尺寸
        is_explicit: 是否为用户明确指定的参数
    
    Returns:
        裁剪后的图像和新的文件名
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    original_shape = image.shape
    height, width = original_shape[:2]
    
    # 计算裁剪坐标
    crop_left, crop_top, crop_right, crop_bottom = calculate_crop_coordinates(crop_params, width, height, is_explicit)
    
    # 裁剪图像
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
    
    # 确定目标尺寸
    if target_size is None:
        target_width, target_height = original_shape[1], original_shape[0]
    else:
        target_width, target_height = target_size
    
    # 缩放到目标尺寸，使用更好的插值方法提高清晰度
    resized_image = cv2.resize(cropped_image, (target_width, target_height), 
                              interpolation=cv2.INTER_CUBIC)
    
    # 生成新的文件名
    image_path_obj = Path(image_path)
    new_filename = f"{image_path_obj.stem}{suffix}{image_path_obj.suffix}"
    
    return resized_image, new_filename


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='图像裁剪和标注文件处理工具')
    parser.add_argument('--image_path', type=str, required=True, help='原图像文件夹路径')
    parser.add_argument('--json_path', type=str, required=True, help='labelme标注文件文件夹路径')
    parser.add_argument('--videocrop', type=str, required=False, nargs='+', default=None,
                       help='裁剪参数, 格式为 left=100 top=200 right=380 bottom=298。如果不提供, 默认裁剪左上角1/4区域')
    parser.add_argument('--target_size', type=str, required=False, default=None, nargs='+',
                       help='目标图像尺寸, 格式为 width=640 height=480。如果不提供, 默认使用原图像尺寸')
    parser.add_argument('--suffix', type=str, default='_cropped', help='裁剪后图像的文件名后缀, 默认为 _cropped。可以设置为空字符串 "" 以保持原文件名')
    parser.add_argument('--output_image_dir', type=str, default='./cropped_images', 
                       help='裁剪后图像保存目录, 默认为 ./cropped_images')
    parser.add_argument('--output_json_dir', type=str, default='./cropped_jsons', 
                       help='裁剪后标注文件保存目录, 默认为 ./cropped_jsons')
    
    args = parser.parse_args()
    
    # 解析裁剪参数
    is_explicit = False
    if args.videocrop is not None:
        crop_params = parse_crop_params(' '.join(args.videocrop))
        is_explicit = True  # 用户明确指定了参数
        print(f"使用自定义裁剪参数: {crop_params}")
    else:
        # 默认裁剪左上角1/4区域
        crop_params = {'left': 0, 'top': 0, 'right': 0, 'bottom': 0}
        is_explicit = False  # 使用默认值
        print("使用默认裁剪参数: 左上角1/4区域")
    
    # 解析目标尺寸参数
    target_size = None
    if args.target_size is not None:
        target_size = parse_target_size(' '.join(args.target_size))
        print(f"使用目标尺寸: {target_size[0]}x{target_size[1]}")
    else:
        print("使用原图像尺寸作为目标尺寸")
    
    # 创建输出目录
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_json_dir, exist_ok=True)
    
    # 获取图像文件列表
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(args.image_path).glob(f'*{ext}'))
        image_files.extend(Path(args.image_path).glob(f'*{ext.upper()}'))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 使用进度条处理每个图像文件
    processed_count = 0
    error_count = 0
    
    for image_file in tqdm(image_files, desc="处理图像", unit="张"):
        try:
            # 裁剪图像并缩放到目标尺寸
            cropped_image, new_filename = crop_image(str(image_file), crop_params, args.suffix, target_size, is_explicit)
            
            # 保存裁剪后的图像
            output_image_path = os.path.join(args.output_image_dir, new_filename)
            cv2.imwrite(output_image_path, cropped_image)
            
            # 查找对应的标注文件
            json_file = Path(args.json_path) / f"{image_file.stem}.json"
            if json_file.exists():
                # 处理标注文件 - 使用裁剪后的图像尺寸
                cropped_shape = cropped_image.shape
                original_shape = cv2.imread(str(image_file)).shape
                processed_data = process_labelme_json(str(json_file), crop_params, cropped_shape, new_filename, original_shape, cropped_image, target_size, is_explicit)
                
                # 保存处理后的标注文件
                output_json_path = os.path.join(args.output_json_dir, 
                                              f"{Path(new_filename).stem}.json")
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
            processed_count += 1
                
        except Exception as e:
            error_count += 1
            tqdm.write(f"处理文件 {image_file.name} 时出错: {str(e)}")
            continue
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 个文件")
    if error_count > 0:
        print(f"处理失败: {error_count} 个文件")
    print(f"裁剪后的图像保存在: {args.output_image_dir}")
    print(f"处理后的标注文件保存在: {args.output_json_dir}")


if __name__ == '__main__':
    main()
