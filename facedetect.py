#!/usr/bin/env python3
# license removed for brevity
import rospy
from sensor_msgs.msg import CompressedImage
import io
import time
import picamera
from edgetpu.detection.engine import DetectionEngine
from PIL import Image as PILImage
from PIL import ImageDraw

MODEL = '/home/ubuntu/edgetpu_api/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
RES = (320,320)

class SplitFrames(object):
    def __init__(self):
        self.frame_num = 0
        self.img = None

    def write(self, buf):
        #find frame in the buffer
        if buf.startswith(b'\xff\xd8'):
            # Run inference.abs
            self.img = PILImage.open(io.BytesIO(buf))
            ans = engine.DetectWithImage(self.img, threshold=0.05,
                                         keep_aspect_ratio=True, 
                                         relative_coord=False, top_k=10)
            self.frame_num += 1
            if ans:
                draw = ImageDraw.Draw(self.img)
                for obj in ans:
                    box = obj.bounding_box.flatten().tolist()
                    # Draw a rectangle.
                    draw.rectangle(box, outline='blue')
                #Publish message
                msg = CompressedImage() 
                msg.header.seq = self.frame_num
                msg.format = 'jpeg'
                msg.header.stamp = rospy.Time.now()
                byteIO = io.BytesIO()
                self.img.save(byteIO, format='JPEG')
                msg.data = byteIO.getvalue()
                pub.publish(msg)

def main():
    global pub
    pub = rospy.Publisher('/FaceDetect/image_raw/compressed', CompressedImage,
                            queue_size=1)
    rospy.init_node('facedetect', anonymous=True)
    rate = rospy.Rate(30) # 30hz

    # Initialize engine
    global engine
    engine = DetectionEngine(MODEL)

    with picamera.PiCamera(resolution=RES, framerate=30) as camera:
        camera.start_preview()
        # Give the camera some warm-up time
        time.sleep(2)
        output = SplitFrames()
        global start
        start = time.time()
        camera.start_recording(output, format='mjpeg')
        camera.wait_recording(0)
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Shutting down")
            camera.stop_recording()


if __name__ == '__main__':
    main()
