import os

os.environ['OMP_NUM_THREADS'] = '1'

import glob
import time
import signal
import sys
import warnings
import platform
import shutil
import onnxruntime
from argparse import ArgumentParser, HelpFormatter

import facefusion.choices
import facefusion.globals
from facefusion.processors.frame.core import get_frame_processors_modules, load_frame_processor_module
from facefusion.utilities import get_temp_frame_paths, list_module_names, decode_execution_providers, normalize_padding

onnxruntime.set_default_logger_severity(3)
warnings.filterwarnings('ignore', category = UserWarning, module = 'gradio')
warnings.filterwarnings('ignore', category = UserWarning, module = 'torchvision')

def set_config():
    # misc
    facefusion.globals.skip_download = False
    facefusion.globals.headless = True
    # execution
    facefusion.globals.execution_providers = decode_execution_providers([ 'cuda' ])
    facefusion.globals.execution_thread_count = 16
    facefusion.globals.execution_queue_count = 1
    # face analyser
    facefusion.globals.face_analyser_order = 'left-right'
    facefusion.globals.face_analyser_age = None
    facefusion.globals.face_analyser_gender = None
    facefusion.globals.face_detector_model = 'retinaface'
    facefusion.globals.face_detector_size = '640x640'
    facefusion.globals.face_detector_score = 0.5
    # face selector
    facefusion.globals.face_selector_mode = 'reference'
    facefusion.globals.reference_face_position = 0
    facefusion.globals.reference_face_distance = 0.6
    facefusion.globals.reference_frame_number = 0
    # face mask
    facefusion.globals.face_mask_blur = 0.3
    facefusion.globals.face_mask_padding = normalize_padding([ 0, 0, 0, 0 ])
    # output creation
    facefusion.globals.output_image_quality = 100
    # frame processors
    available_frame_processors = list_module_names(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facefusion/processors/frame/modules'))
    for frame_processor in available_frame_processors:
        frame_processor_module = load_frame_processor_module(frame_processor)
        if frame_processor == 'face_debugger':
            frame_processor_module.apply_args([ 'kps', 'face-mask' ])
        elif frame_processor == 'face_enhancer':
            frame_processor_module.apply_args('gpen_bfr_512', 100)
        elif frame_processor == 'face_swapper':
            frame_processor_module.apply_args('inswapper_128')

def process_img_dir(input_path):
    set_config()
    # frame_processor_module = get_frame_processors_modules(['face_enhancer'])[0]
    # input_img_list = sorted(os.listdir(input_path))
    # for img_name in input_img_list:
    #     input_img_path = os.path.join(input_path, img_name)
    #     output_img_path = os.path.join(output_path, img_name)

    #     frame_processor_module.process_image(None, input_img_path, output_img_path)
    #     frame_processor_module.post_process()

    temp_frame_paths = sorted(glob.glob(os.path.join(input_path, '*.png')))
    frame_processor_module = get_frame_processors_modules(['face_enhancer'])[0]
    frame_processor_module.process_video(None, temp_frame_paths)
    frame_processor_module.post_process()

if __name__ == '__main__':
    process_img_dir('./input/tmp/contrast_enhanced')