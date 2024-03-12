from ultralytics import YOLO
import sys
import time
import warnings
import git
from pathlib import Path
import pandas as pd
import os
import torch
import numpy as np
from datetime import datetime
import pytz

date_time = str(datetime.now().strftime("%Y%m%d_%H%M"))
jerusalem_timezone = 'Asia/Jerusalem'
date_time_jerusalem = datetime.now(pytz.timezone(jerusalem_timezone)).strftime("%Y%m%d_%H%M")


def save_git_log(output_dir: str, max_commits: int = 3):
    """
    save git log and diff to output_dir. make sure the encoding is utf-8 with no bad characters
    also save git diff, to show what changed since the commit
    :param output_dir:
    :param max_commits:
    :return:
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / 'git_log.txt', 'w', encoding='utf-8') as fp:
        fp.write(git.Repo((str(Path(__file__).parent)),
                          search_parent_directories=True).git.log(max_count=max_commits))
        print(f'git log added to {output_dir}')

    # save git diff
    with open(Path(output_dir) / 'git_diff.txt', 'w', encoding='utf-8') as fp:
        fp.write(git.Repo((str(Path(__file__).parent)),
                          search_parent_directories=True).git.diff())
        print(f'git diff added to {output_dir}')


def train_detector():
    # model = YOLO('yolov8n-LeakyReLU.yaml')  # build a new model from YAML
    model_ckpt = r"/home/ubuntu/multi/ultralytics/runs/detect/20240312_1121_trainDetector_NoKeyboardLightDark_3epochs/weights/best.pt"
    model_yaml = 'yolov8n-LeakyReLU_single_cls.yaml'
    # model = YOLO(model_yaml)  # build a new model from YAML
    model = YOLO(model_yaml).load(model_ckpt)  # build from YAML and transfer weights
    # data_path = 'coco128.yaml'
    data_path = r'/home/ubuntu/data/V7_tagged/spacetop_ea_all_light_NoKeyboardNew_oneFolder.yaml'
    # test name is taken from the data_path
    epochs = 1
    test_name = f'trainDetector_NoKeyboardLightDark_{epochs}epochs'  # Path(data_path).stem
    test_video = r"/home/ubuntu/data/AP_data/20230330_TomerFlight1/1322_20230330_TomerFlight_Dark_Walking/LeftCam/Left.mp4"

    # results_on_test_video_before_train = model(test_video)
    # for i, r in enumerate(results_on_test_video_before_train):
    #     # save as mp4
    #     print('hi')

    # Train the model
    imgsz = [256, 320]
    results = model.train(data=data_path, epochs=epochs, imgsz=[imgsz[0], imgsz[1]], device=0, project='YOLOv8_v1',
                          batch=32, save_period=1, save_json=True, save_yaml=True, save_weights=True, single_cls=True)

    # results_on_test_video_after_train = model(test_video)

    model.info()
    model.export(format='onnx', opset=11, imgsz=imgsz)  # opset_version=11
    # create an example input to export the model

    # data = torch.rand(1, 3, 256, 320)
    # Create a PyTorch tensor with float data
    # data = torch.tensor([1.23, 4.56, 7.89], dtype=torch.float32)
    # # Convert the tensor to a NumPy array and save it to a binary file
    # with open(r"/home/ubuntu/multi/snpe_conversion/20240117_detector_snpe_conversion/real_old_onnx/float_data.bin", 'wb') as file:
    #     np.array(data.numpy(), dtype=np.float32).tofile(file)

    output_results_path = model.trainer.save_dir
    save_git_log(output_results_path)
    
    # change the output_directory to include date and time of the training as a suffix
    output_results_path = Path(output_results_path)
    new_output_results_path = output_results_path.parent / (date_time_jerusalem + '_' + test_name)
    output_results_path.rename(new_output_results_path)
    return model, imgsz


def train_pose(epoch: int = 1):
    # https://docs.ultralytics.com/tasks/pose/#predict
    # Load a model
    model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
    model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

    # Train the model
    imgsz = [640, 480]  # V1
    results = model.train(data='coco8-pose.yaml', epochs=epoch, imgsz=int(np.max(imgsz)))

    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(f'map50-95={metrics.box.map:.2f} map50={metrics.box.map50:.2f} map75={metrics.box.map75:.2f} '
          f'map50-95 of each category={metrics.box.maps}')

    results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    annotated_frame = results[0].plot(boxes=False)

    model.info()
    model.export(format='onnx', opset=11, imgsz=imgsz)  # opset_version=11
    # create an example input to export the model

    # data = torch.rand(1, 3, 256, 320)
    # Create a PyTorch tensor with float data
    # data = torch.tensor([1.23, 4.56, 7.89], dtype=torch.float32)
    # # Convert the tensor to a NumPy array and save it to a binary file
    # with open(r"/home/ubuntu/multi/snpe_conversion/20240117_detector_snpe_conversion/real_old_onnx/float_data.bin", 'wb') as file:
    #     np.array(data.numpy(), dtype=np.float32).tofile(file)

    output_results_path = model.trainer.save_dir
    save_git_log(output_results_path)
    return model, imgsz


def main():
    model, imgsz = train_detector()
    # model, imgsz = train_pose()


if __name__ == '__main__':
    main()
