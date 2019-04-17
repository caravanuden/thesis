import cv2, csv

DATA_DIR = "./raiders/stimuli/task002/scaled_cropped/"
CSV_FILE = "raiders_clips.csv"
STRIDE_LEN = 8

# def get_vid_length(input_video):
#     print(input_video)
#     result = subprocess.Popen('ffprobe -i input_video -show_entries format=duration -v quiet -of csv="p=0"', stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#     output = result.communicate()
#     return output[0]

def get_vid_length(input_video):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)      # Opencv2 version 2 used "CV_CAP_PROP_FPS"
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return n_frames

with open(CSV_FILE, mode="w") as f:
    # write header
    f.write("org_video,label,start_frm,video_id\n")
    id = 0
    for i in range(1,2):
        video = DATA_DIR + "INDIANA_JONES_RAIDERS_LOST_ARK_part_{0}.m4v".format(i)
        for striding in range(0, get_vid_length(video), STRIDE_LEN):
            f.write("{0},0,{1},{2}\n".format(video, striding, id))
            id += 1
