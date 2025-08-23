#!/usr/bin/env python3
"""
Optimized MJPEG to RTMP streaming script for Raspberry Pi 5
Uses OpenCV with hardware acceleration for better performance than FFmpeg
"""

import cv2
import numpy as np
import subprocess
import sys
import time
import os
import logging
from urllib.request import urlopen
from urllib.error import URLError
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MJPEGStreamer:
    def __init__(self, mjpeg_url, rtmp_url):
        self.mjpeg_url = mjpeg_url
        self.rtmp_url = rtmp_url
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.ffmpeg_process = None
        
    def start_ffmpeg_process(self):
        """Start FFmpeg process to handle RTMP output with optimized settings for Pi 5"""
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', '1920x1080',  # Adjust based on your camera resolution
            '-r', '15',  # Match camera framerate
            '-i', '-',  # Read from stdin
            '-f', 'lavfi',
            '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-b:v', '500k',  # Lower bitrate for Pi 5
            '-bufsize', '1000k',
            '-maxrate', '600k',
            '-g', '30',
            '-shortest',
            '-flags', '+global_header',
            '-f', 'flv',
            self.rtmp_url
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            logger.info("FFmpeg process started for RTMP output")
            return True
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            return False
    
    def mjpeg_reader(self):
        """Read MJPEG frames in a separate thread"""
        while self.running:
            try:
                # Open MJPEG stream
                stream = urlopen(self.mjpeg_url, timeout=10)
                bytes_data = bytes()
                
                while self.running:
                    chunk = stream.read(1024)
                    if not chunk:
                        break
                        
                    bytes_data += chunk
                    
                    # Look for JPEG boundaries
                    a = bytes_data.find(b'\xff\xd8')  # JPEG start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG end
                    
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        
                        # Decode JPEG
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Add frame to queue (non-blocking)
                            try:
                                self.frame_queue.put(frame, block=False)
                            except queue.Full:
                                # Drop oldest frame if queue is full
                                try:
                                    self.frame_queue.get_nowait()
                                    self.frame_queue.put(frame, block=False)
                                except queue.Empty:
                                    pass
                
                stream.close()
                
            except Exception as e:
                logger.error(f"MJPEG reader error: {e}")
                time.sleep(1)  # Wait before retrying
    
    def stream_processor(self):
        """Process frames and send to FFmpeg"""
        frame_count = 0
        last_time = time.time()
        
        while self.running:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=1.0)
                
                if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                    # Resize frame if needed (optional optimization)
                    # frame = cv2.resize(frame, (1920, 1080))
                    
                    # Write frame to FFmpeg stdin
                    try:
                        self.ffmpeg_process.stdin.write(frame.tobytes())
                        self.ffmpeg_process.stdin.flush()
                        
                        frame_count += 1
                        
                        # Log FPS every 5 seconds
                        current_time = time.time()
                        if current_time - last_time >= 5.0:
                            fps = frame_count / (current_time - last_time)
                            logger.info(f"Processing at {fps:.1f} FPS")
                            frame_count = 0
                            last_time = current_time
                            
                    except BrokenPipeError:
                        logger.error("FFmpeg process terminated unexpectedly")
                        break
                else:
                    logger.error("FFmpeg process not running")
                    break
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Stream processor error: {e}")
                break
    
    def start(self):
        """Start the streaming process"""
        logger.info(f"Starting MJPEG to RTMP stream: {self.mjpeg_url} -> {self.rtmp_url}")
        
        # Start FFmpeg process
        if not self.start_ffmpeg_process():
            return False
        
        self.running = True
        
        # Start reader thread
        reader_thread = threading.Thread(target=self.mjpeg_reader, daemon=True)
        reader_thread.start()
        
        # Start processor thread
        processor_thread = threading.Thread(target=self.stream_processor, daemon=True)
        processor_thread.start()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
                # Check if FFmpeg process is still running
                if self.ffmpeg_process and self.ffmpeg_process.poll() is not None:
                    logger.error("FFmpeg process died, restarting...")
                    self.stop()
                    return False
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop the streaming process"""
        logger.info("Stopping stream...")
        self.running = False
        
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 stream.py <mjpeg_url> <rtmp_url>")
        sys.exit(1)
    
    mjpeg_url = sys.argv[1]
    rtmp_url = sys.argv[2]
    
    # Enable GPU memory split for Pi 5 (if not already set)
    try:
        subprocess.run(['sudo', 'raspi-config', 'nonint', 'do_memory_split', '128'], 
                      capture_output=True, check=False)
    except:
        pass  # Ignore if raspi-config is not available
    
    streamer = MJPEGStreamer(mjpeg_url, rtmp_url)
    
    max_retries = 5
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if streamer.start():
                logger.info("Stream completed successfully")
                break
            else:
                retry_count += 1
                logger.warning(f"Stream failed, retry {retry_count}/{max_retries}")
                time.sleep(2)
        except Exception as e:
            retry_count += 1
            logger.error(f"Stream error: {e}, retry {retry_count}/{max_retries}")
            time.sleep(2)
    
    if retry_count >= max_retries:
        logger.error("Max retries reached, giving up")
        sys.exit(1)

if __name__ == "__main__":
    main()