# python main.py --id 4 --mp4_path /root/data/next_level/chunk1.mp4 --song_title next_level --chunk chunk1

import os
import cv2
import torch
import numpy as np
import argparse
import random
import shutil
import json
from glob import glob
import sys

# YOLOv5 경로 추가
yolov5_path = os.path.abspath('yolov5')
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# SAM2 경로 및 임포트
from sam2.build_sam import build_sam2_video_predictor

def generate_bright_color(seed):
    random.seed(seed)
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    return [r, g, b]

def show_mask_on_frame(mask, frame, color):
    for c in range(3):
        frame[:, :, c] = np.where(mask > 0, frame[:, :, c] * 0.5 + color[c] * 0.5, frame[:, :, c])
    return frame

def detect_people(frame, yolo_model, device):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    img = img.unsqueeze(0)

    pred = yolo_model(img)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0])

    people_boxes = []
    
    if pred[0] is not None and len(pred[0]) > 0:
        det = pred[0]
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
        
        for *xyxy, conf, cls in det:
            xyxy = list(map(int, xyxy))
            people_boxes.append(xyxy)
    else:
        print("No people detected in this frame.")

    return people_boxes

def draw_prompt_points(frame, points):
    for point in points:
        cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
    return frame

def save_last_frame_mask(video_segments, frame_idx, frame, output_dir, obj_ids):
    """Save the mask for the last frame and visualize the mask on the frame."""
    frame_masks = video_segments.get(frame_idx, {})
    for obj_id, mask in frame_masks.items():
        mask_np = mask.cpu().numpy()  # Convert tensor to numpy
        color = generate_bright_color(obj_id)
        frame = show_mask_on_frame(mask_np, frame, color)
    cv2.imwrite(f"{output_dir}/last_frame_{frame_idx}_with_masks.jpg", frame)
    # print(f"Saved mask visualization for frame {frame_idx}.")

from tqdm import tqdm

def save_masks_to_json(video_segments, json_output_dir):
    """각 프레임의 마스크 정보를 JSON 파일로 저장합니다."""
    os.makedirs(json_output_dir, exist_ok=True)
    # tqdm을 사용하여 진행 상황 표시
    for frame_idx, masks in tqdm(video_segments.items(), total=len(video_segments), desc='Saving masks'):
        mask_info = {}
        for obj_id, mask in masks.items():
            mask_np = mask.cpu().numpy().tolist()  # JSON으로 저장하기 위해 리스트로 변환
            mask_info[obj_id] = mask_np
        json_file_path = os.path.join(json_output_dir, f"frame_{frame_idx}_masks.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(mask_info, json_file)
        # print 문은 제거합니다.
        # print(f"Saved mask data for frame {frame_idx} to {json_file_path}.")

# Argument parsing
parser = argparse.ArgumentParser(description='SAM2 with YOLOv5 person detection')
parser.add_argument('--id', type=int, default=6, help='Object ID to match with detected people')
parser.add_argument('--mp4_path', type=str, required=True, help='Path to input MP4 video')
parser.add_argument('--song_title', type=str, required=True, help='Song title to organize results folders')
parser.add_argument('--chunk', type=str, required=True, help='Chunk Number (like chunk1)')
args = parser.parse_args()
ann_obj_id = args.id
mp4_path = args.mp4_path
song_title = args.song_title
chunk_num = args.chunk

# temp_frame 폴더 생성
temp_frame_dir = "./temp_frame"
if os.path.exists(temp_frame_dir):
    shutil.rmtree(temp_frame_dir)
os.makedirs(temp_frame_dir)

def split_video_into_frames(video_path, output_dir, desired_fps=30):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Original Video FPS: {original_fps}, Total Frames: {frame_count}")

    # FPS를 30으로 고정
    fps = desired_fps
    frame_interval = int(original_fps // fps)  # 얼마나 자주 프레임을 저장할지 결정

    current_frame = 0
    saved_frames = 0
    chunk_idx = 1
    saved_frames_in_chunk = 0
    min_chunk_size = 500
    max_chunk_size = 1000
    chunk_lengths = []  # To store the length of each chunk

    chunk_folder = os.path.join(output_dir, str(chunk_idx))
    os.makedirs(chunk_folder, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # FPS 맞추기 위해 일정한 간격으로 프레임 저장
        if current_frame % frame_interval == 0:
            frame_filename = f"{saved_frames_in_chunk:05d}.jpg"
            frame_path = os.path.join(chunk_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frames += 1
            saved_frames_in_chunk += 1

        current_frame += 1

        # 청크 종료 조건: 마지막 프레임에 YOLO로 사람 감지 후, 감지된 인원과 ann_obj_id 비교
        if saved_frames_in_chunk >= min_chunk_size:
            people_boxes = detect_people(frame, yolo_model, device)
            detected_people_count = len(people_boxes)

            if detected_people_count == ann_obj_id:
                chunk_lengths.append(saved_frames_in_chunk)
                saved_frames_in_chunk = 0
                chunk_idx += 1
                chunk_folder = os.path.join(output_dir, str(chunk_idx))
                os.makedirs(chunk_folder, exist_ok=True)
            elif saved_frames_in_chunk >= max_chunk_size:
                raise ValueError(f"Chunk length exceeded the maximum allowed limit of {max_chunk_size} frames.")
            
    cap.release()
    return fps, frame_width, frame_height, chunk_lengths


# YOLOv5 모델 설정
weights = 'yolov5/yolov5s.pt'
device = select_device('')
yolo_model = DetectMultiBackend(weights, device=device)

# SAM2 모델 설정
checkpoint = "./checkpoints/sam2_hiera_small.pt"
model_cfg = "sam2_hiera_s.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 결과 디렉터리 생성
results_dir = os.path.join("/root/results", song_title, chunk_num)
video_segments = {}
chunk_paths = []
prompt_points_output_dir = os.path.join(results_dir, "prompt_points")
output_videos_dir = os.path.join(results_dir, "videos")
json_output_dir = os.path.join(results_dir, "mask_data")

os.makedirs(prompt_points_output_dir, exist_ok=True)
os.makedirs(output_videos_dir, exist_ok=True)

# Step 1: 비디오를 프레임으로 분할 (FPS 30으로 변경)
fps, frame_width, frame_height, chunk_lengths = split_video_into_frames(mp4_path, temp_frame_dir)

# 모델 시작 전 청크 정보 출력
print(f"Total {len(chunk_lengths)} chunks detected.")

# 각 청크를 처리
total_chunks = len(os.listdir(temp_frame_dir))
for chunk_idx in range(1, total_chunks + 1):
    chunk_dir = os.path.join(temp_frame_dir, str(chunk_idx))
    
    frame_names = [p for p in os.listdir(chunk_dir) if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # 첫 번째 프레임 로드
    frame_0 = cv2.imread(os.path.join(chunk_dir, frame_names[0]))
    
    if chunk_idx == 1:
        people_boxes = detect_people(frame_0, yolo_model, device)
        detected_people_count = len(people_boxes)

        # YOLO 감지 결과 출력
        print(f"YOLO detected {detected_people_count} people in frame 0.")
        for idx, box in enumerate(people_boxes):
            print(f"Person {idx + 1}: Bounding box {box}")
        print()

        if detected_people_count != ann_obj_id:
            print(f"Error: Detected {detected_people_count} people but expected {ann_obj_id}. Exiting...")
            exit(1)

        # 프롬프트 생성
        inference_state = predictor.init_state(video_path=chunk_dir)
        prompts = {}
        prompt_points = []
        for idx, box in enumerate(people_boxes):
            obj_id = idx + 1
            points = np.array([[int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)]], dtype=np.float32)
            prompt_points.append(points[0].astype(int))
            labels = np.array([1], np.int32)
            prompts[obj_id] = points, labels
            predictor.add_new_points_or_box(inference_state, frame_idx=0, obj_id=obj_id, points=points, labels=labels)

        # 프롬프트 시각화
        frame_with_prompts = draw_prompt_points(frame_0.copy(), prompt_points)
        cv2.imwrite(f"{prompt_points_output_dir}/frame_{(chunk_idx-1)*1000}_prompts.jpg", frame_with_prompts)
    else:
        # 이전 청크의 마지막 프레임에서 마스크를 가져옴
        last_frame_idx = sum(chunk_lengths[:chunk_idx - 1]) - 1
        masks = video_segments.get(last_frame_idx, {})
        if not masks:
            print(f"Error: No masks found for frame {last_frame_idx}. Exiting...")
            exit(1)
        
        # 다음 청크의 첫 번째 프레임에서 마스크 프롬프트 사용
        inference_state = predictor.init_state(video_path=chunk_dir)
        for obj_id, mask in masks.items():
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if mask.dim() > 2:
                mask = mask.squeeze()
            mask = mask.cpu()
            mask = mask.to(dtype=torch.bool)
            predictor.add_new_mask(inference_state, frame_idx=0, obj_id=obj_id, mask=mask)

        print(f"Loaded masks from frame {last_frame_idx} as prompts for chunk {chunk_idx}.")

    # SAM2 전파 실행
    print(f"Starting SAM2 propagation for chunk {chunk_idx}...")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if len(out_obj_ids) > 0:
            global_frame_idx = sum(chunk_lengths[:chunk_idx - 1]) + out_frame_idx
            video_segments[global_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if out_frame_idx % 500 == 0:
                print(f"Propagation at frame {global_frame_idx} for object IDs {out_obj_ids}.")
        else:
            print(f"No object IDs found at frame {global_frame_idx}.")

        if out_frame_idx % 500 == 0:
            torch.cuda.empty_cache()
            print(f"Cleared GPU cache after processing frame {out_frame_idx}.")

    # 마지막 프레임의 마스크를 시각화하고 저장
    last_frame_path = os.path.join(chunk_dir, frame_names[-1])
    last_frame = cv2.imread(last_frame_path)
    save_last_frame_mask(video_segments, sum(chunk_lengths[:chunk_idx]) - 1, last_frame, prompt_points_output_dir, out_obj_ids)

    # 비디오 저장
    output_video_path = f'{output_videos_dir}/output_video_chunk_{chunk_idx}.mp4'
    chunk_paths.append(output_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for out_frame_idx in range(len(frame_names)):
        global_frame_idx = sum(chunk_lengths[:chunk_idx - 1]) + out_frame_idx
        frame = cv2.imread(os.path.join(chunk_dir, frame_names[out_frame_idx]))
        if global_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[global_frame_idx].items():
                color = generate_bright_color(out_obj_id)
                out_mask_np = out_mask.numpy()
                frame = show_mask_on_frame(out_mask_np, frame, color)
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved to {output_video_path}")

    # 상태 초기화 및 메모리 정리
    predictor.reset_state(inference_state)
    del inference_state
    torch.cuda.empty_cache()

    print(f"Finished processing chunk {chunk_idx}. Resetting state and moving to next chunk.\n\n")

# 모든 청크 비디오를 결합
final_output_video = f'{output_videos_dir}/final_output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
final_video_writer = cv2.VideoWriter(final_output_video, fourcc, fps, (frame_width, frame_height))

for chunk_path in chunk_paths:
    cap = cv2.VideoCapture(chunk_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        final_video_writer.write(frame)
    cap.release()

final_video_writer.release()
print(f"Final video saved to {final_output_video}")

# 마스크 데이터 저장
save_masks_to_json(video_segments, json_output_dir)

# temp_frame 디렉터리 삭제
shutil.rmtree(temp_frame_dir)
print(f"Deleted temp_frame directory after processing.")
