import av
import os
import sys
import streamlit as st
import cv2
import tempfile


BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)


from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner, get_thresholds_pro



st.title('AI Fitness Trainer: Squats Analysis')

mode = st.radio('Select Mode', ['Beginner', 'Pro'], horizontal=True)



thresholds = None 

if mode == 'Beginner':
    thresholds = get_thresholds_beginner()

elif mode == 'Pro':
    thresholds = get_thresholds_pro()



upload_process_frame = ProcessFrame(thresholds=thresholds)

# Initialize face mesh solution
pose = get_mediapipe_pose()


output_video_file = f'output_recorded.mp4'

if os.path.exists(output_video_file):
    os.remove(output_video_file)


with st.form('Upload', clear_on_submit=True):
    up_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'])
    uploaded = st.form_submit_button("Upload")

stframe = st.empty()

ip_vid_str = '<p style="font-family:Helvetica; font-weight: bold; font-size: 16px;">Input Video</p>'
warning_str = '<p style="font-family:Helvetica; font-weight: bold; color: Red; font-size: 17px;">Please Upload a Video first!!!</p>'

warn = st.empty()


download_button = st.empty()

if up_file and uploaded:
    
    download_button.empty()
    tfile = tempfile.NamedTemporaryFile(delete=False)

    try:
        warn.empty()
        tfile.write(up_file.read())

        vf = cv2.VideoCapture(tfile.name)
        vf_av = av.open(tfile.name)

        # ---------------------  Write the processed video frame. --------------------
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)
        # -----------------------------------------------------------------------------

        
        txt = st.sidebar.markdown(ip_vid_str, unsafe_allow_html=True)   
        ip_video = st.sidebar.video(tfile.name) 

        for frame in vf_av.decode(video=0):

            # convert frame from BGR to RGB before processing it.
            frame_rgb = frame.to_ndarray(format="rgb24")
            # out_frame, _ = upload_process_frame.process(frame_rgb, pose)
            stframe.image(frame_rgb)
            video_output.write(frame_rgb[...,::-1])

        
        vf.release()
        video_output.release()
        stframe.empty()
        ip_video.empty()
        txt.empty()
        tfile.close()
    
    except AttributeError:
        warn.markdown(warning_str, unsafe_allow_html=True)   



if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download_button.download_button('Download Video', data = op_vid, file_name='output_recorded.mp4')







    
    

    


