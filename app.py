import uuid
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

import av
import cv2
import streamlit as st
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer

def upload_to_s3(file_path: Path, bucket: str, s3_key: str = None):
    """Upload file to S3 bucket using boto3."""
    if s3_key is None:
        s3_key = file_path.name
    
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(str(file_path), bucket, s3_key)
        return True, f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    except ClientError as e:
        st.error(f"S3 upload failed: {e}")
        return False, None

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def app():
    # S3 configuration - replace with your bucket
    BUCKET_NAME = "your-streamlit-videos-2025"
    
    if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())
    prefix = st.session_state["prefix"]
    RECORD_DIR = Path("./records")
    RECORD_DIR.mkdir(exist_ok=True)
    
    in_file = RECORD_DIR / f"{prefix}_input.flv"
    out_file = RECORD_DIR / f"{prefix}_output.flv"
    
    # S3 keys with timestamp prefix for uniqueness
    timestamp = st.session_state["prefix"][:8]  # First 8 chars of UUID
    in_s3_key = f"recordings/{timestamp}_input.flv"
    out_s3_key = f"recordings/{timestamp}_output.flv"

    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(str(in_file), format="flv")

    def out_recorder_factory() -> MediaRecorder:
        return MediaRecorder(str(out_file), format="flv")

    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": True},
        video_frame_callback=video_frame_callback,
        in_recorder_factory=in_recorder_factory,
        out_recorder_factory=out_recorder_factory,
    )

    # Handle input file
    if in_file.exists():
        success, s3_url = upload_to_s3(in_file, BUCKET_NAME, in_s3_key)
        if success:
            st.success(f"Input video uploaded to S3: [Download]({s3_url})")
            st.video(s3_url)  # Preview in Streamlit
            in_file.unlink()  # Clean up local file
        else:
            with in_file.open("rb") as f:
                st.download_button("Download input (local)", f, "input.flv")

    # Handle output file  
    if out_file.exists():
        success, s3_url = upload_to_s3(out_file, BUCKET_NAME, out_s3_key)
        if success:
            st.success(f"Output video uploaded to S3: [Download]({s3_url})")
            st.video(s3_url)  # Preview in Streamlit
            out_file.unlink()  # Clean up local file
        else:
            with out_file.open("rb") as f:
                st.download_button("Download output (local)", f, "output.flv")

if __name__ == "__main__":
    app()
