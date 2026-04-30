# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s-seg.pt                 # PyTorch
                                          yolov5s-seg.torchscript        # TorchScript
                                          yolov5s-seg.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                          yolov5s-seg_openvino_model     # OpenVINO
                                          yolov5s-seg.engine             # TensorRT
                                          yolov5s-seg.mlmodel            # CoreML (macOS-only)
                                          yolov5s-seg_saved_model        # TensorFlow SavedModel
                                          yolov5s-seg.pb                 # TensorFlow GraphDef
                                          yolov5s-seg.tflite             # TensorFlow Lite
                                          yolov5s-seg_edgetpu.tflite     # TensorFlow Edge TPU
                                          yolov5s-seg_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
    source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_conf=False,  # save confidences in labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    nosaveimage=False,  # do not save images
    nosavejson=False,  # do not save json files
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / 'runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    image_name='images',  # directory name for saving images
    json_name='json',  # directory name for saving json files
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):
    source = str(source)
    save_img = not nosave and not nosaveimage and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    if not nosavejson:
        (save_dir / json_name).mkdir(parents=True, exist_ok=True)  # make json dir
    if not nosaveimage:
        (save_dir / image_name).mkdir(parents=True, exist_ok=True)  # make image dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    # ==================== 模型信息 ====================
    LOGGER.info("🚀 模型信息")
    LOGGER.info("=" * 50)
    
    # 模型权重信息
    weights_str = f'📁 模型权重: {str(weights)}'
    LOGGER.info(weights_str)
    
    # 设备信息
    LOGGER.info(f'🔧 设备: {device}')
    
    # 推理尺寸
    LOGGER.info(f'📐 推理尺寸: {imgsz}')
    
    # 类别数量
    LOGGER.info(f'🏷️  类别数量: {len(names)}')
    
    # 有效类别ID范围
    LOGGER.info(f'🔢 有效类别ID范围: 0-{len(names)-1}')
    
    # 类别列表
    LOGGER.info('📋 类别列表:')
    for i, (key, value) in enumerate(names.items()):
        LOGGER.info(f'    {key}: "{value}"')
    
    LOGGER.info("")

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # ==================== 开始推理 ====================
    LOGGER.info("🎯 开始推理")
    LOGGER.info("=" * 50)
    
    # 输入源信息
    LOGGER.info(f'📂 输入源: {str(source)}')
    
    # 其他参数
    LOGGER.info(f'🎚️  置信度阈值: {conf_thres}')
    LOGGER.info(f'🔗 IoU阈值: {iou_thres}')
    LOGGER.info(f'📊 最大检测数量: {max_det}')
    LOGGER.info("")
    
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    # 初始化类别验证统计
    valid_detections = 0
    invalid_detections = 0
    images_with_detections = 0  # 新增：跟踪包含有效检测的图像数量
    class_counts = {}
    # 确保class_counts字典包含所有可能的类别名称
    for i, name in enumerate(names):
        class_counts[name] = 0
    
    # 创建进度条
    total_files = len(dataset) if hasattr(dataset, '__len__') else None
    pbar = tqdm(dataset, total=total_files, desc="🚀 推理进度", unit="张", ncols=120)
    
    for path, im, im0s, vid_cap, s in pbar:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / image_name / p.name)  # im.jpg
            json_path = str(save_dir / json_name / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.json
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            # Initialize JSON data structure
            if not nosavejson:
                json_data = {
                    "version": "5.2.1",
                    "flags": {},
                    "shapes": [],
                    "imagePath": p.name,
                    "imageData": None,
                    "imageHeight": im0.shape[0],
                    "imageWidth": im0.shape[1]
                }
            
            if len(det):
                # 标记当前图像包含检测
                images_with_detections += 1
                
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments - always compute for consistency
                segments = masks2segments(masks)
                segments = [scale_segments(im.shape[2:], x, im0.shape, normalize=True) for x in segments]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(masks,
                                colors=[colors(x, True) for x in det[:, 5]],
                                im_gpu=None if retina_masks else im[i])

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(det[:, :6]):  # Use original order, not reversed
                    # 第一步：验证类别ID是否在有效范围内
                    try:
                        cls_id = int(cls)
                        if cls_id < 0 or cls_id >= len(names):
                            # LOGGER.warning(f"跳过无效类别ID: {cls_id}, 有效范围: 0-{len(names)-1}")
                            invalid_detections += 1
                            continue
                        
                        # 获取类别名称并验证
                        class_name = names[cls_id]
                        if class_name is None or class_name == "":
                            # LOGGER.warning(f"跳过空类别名称，类别ID: {cls_id}")
                            invalid_detections += 1
                            continue
                        
                    except (ValueError, IndexError, TypeError) as e:
                        # LOGGER.warning(f"类别ID处理错误: {cls}, 错误信息: {e}")
                        invalid_detections += 1
                        continue
                    
                    # 第二步：验证segments是否存在且有效
                    if j >= len(segments):
                        # LOGGER.warning(f"跳过无效的segment索引: {j}, 总segments数量: {len(segments)}")
                        invalid_detections += 1
                        continue
                    
                    segj = segments[j].reshape(-1)  # (n,2) to (n*2)
                    
                    # 第三步：验证坐标值
                    try:
                        segj_valid = [float(x) for x in segj]
                        
                        # 检查坐标值数量是否为偶数（每个点需要x,y两个坐标）
                        if len(segj_valid) % 2 != 0:
                            # LOGGER.warning(f"跳过坐标值数量不正确的检测: {len(segj_valid)} 个值")
                            invalid_detections += 1
                            continue
                        
                        # 检查坐标值是否在有效范围内
                        coordinates_valid = True
                        for coord_idx in range(0, len(segj_valid), 2):
                            x, y = segj_valid[coord_idx], segj_valid[coord_idx + 1]
                            if not (0 <= x <= 1) or not (0 <= y <= 1):
                                # LOGGER.warning(f"跳过无效的坐标值: [{x}, {y}], 坐标索引: {coord_idx//2}")
                                invalid_detections += 1
                                coordinates_valid = False
                                break
                        
                        if not coordinates_valid:
                            continue
                        
                        # 检查是否有足够的坐标点（至少3个点形成多边形）
                        if len(segj_valid) < 6:  # 至少需要3个点的坐标(6个值)
                            # LOGGER.warning(f"跳过坐标点不足的检测: {len(segj_valid)} 个值，需要至少6个")
                            invalid_detections += 1
                            continue
                        
                        # 所有验证都通过，现在可以安全地写入文件
                        
                        # 更新统计信息
                        valid_detections += 1
                        if class_name not in class_counts:
                            class_counts[class_name] = 0
                        class_counts[class_name] += 1
                        
                        # 写入JSON文件
                        if not nosavejson:
                            # Convert normalized coordinates to absolute coordinates
                            segj_abs = masks2segments(masks[j:j+1])[0]  # Get segment points
                            segj_abs = scale_segments(im.shape[2:], segj_abs, im0.shape, normalize=False)  # Scale to image size
                            
                            # Convert points to list format
                            points = [[float(x), float(y)] for x, y in segj_abs]
                            
                            # Only save if points are not empty and have at least 3 points for polygon
                            if len(points) >= 3:
                                shape_data = {
                                    "mask": None,
                                    "label": names[cls_id],  # 使用验证后的cls_id
                                    "points": points,
                                    "group_id": None,
                                    "description": "",
                                    "shape_type": "polygon",
                                    "flags": {}
                                }
                                json_data["shapes"].append(shape_data)
                            
                            # 图像标注
                            if save_img or save_crop or view_img:
                                c = cls_id  # 使用验证后的cls_id
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            
                            # 保存裁剪图像
                            if save_crop:
                                c = cls_id
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            
                    except (ValueError, TypeError) as e:
                        # LOGGER.warning(f"坐标值处理错误: {segj}, 错误信息: {e}")
                        invalid_detections += 1
                        continue

            # Save JSON file
            if not nosavejson:
                with open(f'{json_path}.json', 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 更新进度条信息
        if len(det) > 0:
            det_info = f"🎯 {len(det)} 个目标"
        else:
            det_info = "❌ 无目标"
        
        # 限制文件名长度，避免进度条显示问题
        filename = p.name
        if len(filename) > 25:
            filename = filename[:22] + "..."
        
        pbar.set_postfix({
            '📁': filename,
            '🔍': det_info,
            '⏱️': f"{dt[1].dt * 1E3:.1f}ms"
        }, refresh=True)

    # 计算统计信息
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    total_detections = valid_detections + invalid_detections
    detected_classes = [(name, count) for name, count in class_counts.items() if count > 0]
    detected_classes.sort(key=lambda x: x[1], reverse=True)  # 按检测数量排序
    
    # 文件统计
    if not nosavejson:
        json_files = list(save_dir.glob(f'{json_name}/*.json'))
        json_count = len(json_files)
    else:
        json_count = 0
    
    if not nosaveimage:
        image_files = list(save_dir.glob(f'{image_name}/*.jpg')) + list(save_dir.glob(f'{image_name}/*.png'))
        image_count = len(image_files)
    else:
        image_count = 0
    
    total_files = json_count + image_count
    
    # ==================== 推理性能统计 ====================
    LOGGER.info("⚡ 推理性能统计")
    LOGGER.info("=" * 50)
    LOGGER.info(f'🖼️  处理图像数量: {seen}')
    LOGGER.info(f'⏱️  平均预处理时间: {t[0]:.1f}ms')
    LOGGER.info(f'🚀 平均推理时间: {t[1]:.1f}ms')
    LOGGER.info(f'🔍 平均NMS时间: {t[2]:.1f}ms')
    LOGGER.info(f'📐 推理尺寸: {(1, 3, *imgsz)}')
    LOGGER.info("")
    
    # ==================== 类别验证统计 ====================
    LOGGER.info("✅ 类别验证统计")
    LOGGER.info("=" * 50)
    LOGGER.info(f'📊 总检测数量: {total_detections}')
    LOGGER.info(f'✅ 有效检测数量: {valid_detections}')
    LOGGER.info(f'❌ 无效检测数量: {invalid_detections}')
    if total_detections > 0:
        valid_rate = (valid_detections / total_detections) * 100
        invalid_rate = (invalid_detections / total_detections) * 100
        LOGGER.info(f'📈 有效检测率: {valid_rate:.1f}%')
        if invalid_detections > 0:
            LOGGER.warning(f'⚠️  无效检测率: {invalid_rate:.1f}%')
    LOGGER.info("")
    
    # ==================== 各类别检测统计 ====================
    LOGGER.info("🏷️  各类别检测统计")
    LOGGER.info("=" * 50)
    if detected_classes:
        for i, (class_name, count) in enumerate(detected_classes):
            percentage = (count / valid_detections * 100) if valid_detections > 0 else 0
            LOGGER.info(f'  {class_name:<25} {count:>6} 个 ({percentage:>5.1f}%)')
    else:
        LOGGER.info('  未检测到任何有效目标')
    LOGGER.info("")
    
    # ==================== 文件保存统计 ====================
    LOGGER.info("💾 文件保存统计")
    LOGGER.info("=" * 50)
    
    # 保存目录信息
    LOGGER.info(f'📁 保存目录: {str(save_dir)}')
    
    # JSON文件统计
    if not nosavejson:
        LOGGER.info(f'📄 JSON标注文件: {json_count:>6} 个')
        LOGGER.info(f'   -> {str(save_dir / json_name)}')
    else:
        LOGGER.info('📄 JSON标注文件: 已禁用保存')
    
    # 图像文件统计
    if not nosaveimage:
        LOGGER.info(f'🖼️  检测结果图像: {image_count:>6} 个')
        LOGGER.info(f'   -> {str(save_dir / image_name)}')
    else:
        LOGGER.info('🖼️  检测结果图像: 已禁用保存')
    LOGGER.info("")
    
    # ==================== 执行总结 ====================
    LOGGER.info("🎉 执行总结")
    LOGGER.info("=" * 50)
    
    # 模型权重信息
    LOGGER.info(f'📁 模型权重: {str(weights)}')
    
    # 输入源信息
    LOGGER.info(f'📂 输入源: {str(source)}')
    
    # 其他参数
    LOGGER.info(f'🎚️  置信度阈值: {conf_thres}')
    LOGGER.info(f'🔗 IoU阈值: {iou_thres}')
    LOGGER.info(f'🔧 设备: {device}')
    
    # 计算总体成功率
    if seen > 0:
        success_rate = (images_with_detections / seen) * 100 if images_with_detections > 0 else 0
        LOGGER.info(f'📈 总体成功率: {success_rate:.1f}%')
        LOGGER.info(f'   ({images_with_detections}/{seen} 图像包含有效检测)')
    
    # 文件保存总结
    LOGGER.info(f'📊 生成文件总数: {total_files} 个')
    
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
        LOGGER.info('🔄 模型已更新')
    
    LOGGER.info("")
    LOGGER.info("🎊 推理完成")
    LOGGER.info("=" * 50)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--nosaveimage', action='store_true', help='do not save images')
    parser.add_argument('--nosavejson', action='store_true', help='do not save json files')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--image-name', default='images', help='directory name for saving images')
    parser.add_argument('--json-name', default='json', help='directory name for saving json files')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
