import json
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


import os
import logging
logger.info("!!!!! Current file path: {}".format(os.path.abspath(__file__)))
logger.info("##### Current dir files: {}".format(os.listdir()))

import torch
from torch.utils.data import DataLoader
from C3D_altered import C3D_altered
from my_fc6 import my_fc6
from score_regressor import score_regressor
# from opts import *
import numpy as np
import cv2 as cv
import tempfile
from torchvision import transforms
import base64
import boto3
import json

def lambda_handler(event, context):
    try:
        # logger.info(event)
        # suvid = "video"
        suvid = "body"
        # video = event["queryStringParameters"]['video']
        logger.info("suhyun: input received")
        # logger.info(f"suhyun event looks like {event}")
        # logger.info(f"suhyun event[video] looks like {event[suvid]}")
        video = event[suvid]
        # video = event["video"]

        # ## Trying to see if S3 fetching works (yes)
        # # Creating the low level functional client
        # client = boto3.client('s3')
        
        # # Fetch the list of existing buckets
        # clientResponse = client.list_buckets()
        
        # # Print the bucket names one by one
        # logger.info('Printing bucket names...')
        # bucketList = []
        # for bucket in clientResponse['Buckets']:
        #     bucketList.append(bucket["Name"])
        #     logger.info(f'Bucket Name: {bucket["Name"]}')
 
    except KeyError:
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "score": "empty",
                }
            ),
        }
    
    frames = preprocess_one_video(video)
    preds = inference_with_one_video_frames(frames)
    val = int(preds[0] * 17)
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "score": val,
            }
        ),
    }


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def center_crop(img, dim):
    """Returns center cropped image

    Args:Image Scaling
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def preprocess_one_video(video_data):
    # TODO: fix this later
    C, H, W = 3,112,112
    input_resize = 171,128
    test_batch_size = 1


    if video_data is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(base64.b64decode(video_data))

        vf = cv.VideoCapture(tfile.name)

        # https: // discuss.streamlit.io / t / how - to - access - uploaded - video - in -streamlit - by - open - cv / 5831 / 8
        frames = None
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            frame = cv.resize(frame, input_resize, interpolation=cv.INTER_LINEAR) #frame resized: (128, 171, 3)
            frame = center_crop(frame, (H, H))
            frame = transform(frame).unsqueeze(0)
            if frames is not None:
                frame = np.vstack((frames, frame))
                frames = frame
            else:
                frames = frame


        vf.release()
        cv.destroyAllWindows()
        rem = len(frames) % 16
        rem = 16 - rem

        if rem != 0:
            padding = np.zeros((rem, C, H, H))
            logger.info(padding.shape)
            frames = np.vstack((frames, padding))

        frames = np.expand_dims(frames, axis=0)
        logger.info(f"frames shape: {frames.shape}")
        # frames shape: (137, 3, 112, 112)
        frames = DataLoader(frames, batch_size=test_batch_size, shuffle=False)
        return frames

def load_weights():
    # TODO: fix this import issue
    m1_path = 'model_CNN_94.pth'
    m2_path = 'model_my_fc6_94.pth'
    m3_path = 'model_score_regressor_94.pth'
    m4_path = 'model_dive_classifier_94.pth'

    current_path = os.path.abspath(os.getcwd())
    m1_path = os.path.join(current_path, m1_path)
    m2_path = os.path.join(current_path, m2_path)
    m3_path = os.path.join(current_path, m3_path)
    m4_path = os.path.join(current_path, m4_path)

    cnn_loaded = os.path.isfile(m1_path)
    fc6_loaded = os.path.isfile(m2_path)
    s_reg_loaded = os.path.isfile(m3_path)
    dive_cla_loaded = os.path.isfile(m4_path)

    if cnn_loaded and fc6_loaded and s_reg_loaded and dive_cla_loaded:
        return

    BUCKET_NAME = 'aqa-diving'
    BUCKET_WEIGHT_FC6 = 'model_my_fc6_94.pth'
    BUCKET_WEIGHT_CNN = 'model_CNN_94.pth'
    BUCKET_WEIGHT_S_REG = 'model_score_regressor_94.pth'
    BUCKET_WEIGHT_DIVE_CLA = 'model_dive_classifier_94.pth'

    s3 = boto3.client('s3')
    if not cnn_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_CNN, m1_path)
    if not fc6_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_FC6, m2_path)
    if not s_reg_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_S_REG, m3_path)
    if not dive_cla_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_DIVE_CLA, m4_path)

def inference_with_one_video_frames(frames):
    m1_path = 'model_CNN_94.pth'
    m2_path = 'model_my_fc6_94.pth'
    m3_path = 'model_score_regressor_94.pth'
    m4_path = 'model_dive_classifier_94.pth'
    with_dive_classification = False
    
    current_path = os.path.abspath(os.getcwd())
    m1_path = os.path.join(current_path, m1_path)
    m2_path = os.path.join(current_path, m2_path)
    m3_path = os.path.join(current_path, m3_path)

    logger.info("Starting to load the trained models...")
    load_weights()

    model_CNN = C3D_altered()
    model_CNN.load_state_dict(torch.load(m1_path, map_location={'cuda:0': 'cpu'}))

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.load_state_dict(torch.load(m2_path, map_location={'cuda:0': 'cpu'}))

    # loading our score regressor
    model_score_regressor = score_regressor()
    model_score_regressor.load_state_dict(torch.load(m3_path, map_location={'cuda:0': 'cpu'}))
    logger.info('Using Final Score Loss')
    with torch.no_grad():
        pred_scores = [];
        # true_scores = []
        if with_dive_classification:
            pred_position = [];
            pred_armstand = [];
            pred_rot_type = [];
            pred_ss_no = [];
            pred_tw_no = []
            true_position = [];
            true_armstand = [];
            true_rot_type = [];
            true_ss_no = [];
            true_tw_no = []

        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()

        for video in frames:
            logger.info(f"video shape: {video.shape}") # video shape: torch.Size([1, 144, 3, 112, 112])
            video = video.transpose_(1, 2)
            video = video.double()
            clip_feats = torch.Tensor([])
            for i in np.arange(0, len(video), 16):
                logger.info(i)
                clip = video[:, :, i:i + 16, :, :]
                logger.info(f"clip shape: {clip.shape}") # clip shape: torch.Size([1, 3, 16, 112, 112])
                logger.info(f"clip type: {clip.type()}") # clip type: torch.DoubleTensor
                model_CNN = model_CNN.double()
                clip_feats_temp = model_CNN(clip)

                logger.info(f"clip_feats_temp shape: {clip_feats_temp.shape}")
                # clip_feats_temp shape: torch.Size([1, 8192])

                clip_feats_temp.unsqueeze_(0)

                logger.info(f"clip_feats_temp unsqueeze shape: {clip_feats_temp.shape}")
                # clip_feats_temp unsqueeze shape: torch.Size([1, 1, 8192])

                clip_feats_temp.transpose_(0, 1)

                logger.info(f"clip_feats_temp transposes shape: {clip_feats_temp.shape}")
                # clip_feats_temp transposes shape: torch.Size([1, 1, 8192])

                clip_feats = torch.cat((clip_feats.double(), clip_feats_temp), 1)

                logger.info(f"clip_feats shape: {clip_feats.shape}")
                # clip_feats shape: torch.Size([1, 1, 8192])

            clip_feats_avg = clip_feats.mean(1)

            logger.info(f"clip_feats_avg shape: {clip_feats_avg.shape}") # clip_feats_avg shape: torch.Size([1, 8192])

            model_my_fc6 = model_my_fc6.double()
            sample_feats_fc6 = model_my_fc6(clip_feats_avg)
            model_score_regressor = model_score_regressor.double()
            temp_final_score = model_score_regressor(sample_feats_fc6)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])

            logger.info('Predicted scores: ', pred_scores)
            return pred_scores
