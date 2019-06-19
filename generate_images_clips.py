import cv2, csv, subprocess

# DATA_DIR = "./raiders/stimuli/task002/scaled_cropped/"
DATA_DIR = '/Users/caravanuden/thesis/life/scaled_cropped/'
IMG_DIR = '/Users/caravanuden/thesis/life/images/'
CSV_FILE = 'raiders_clips.csv'
CLIP_LEN = 32
IMAGE_SAMP = 15
RUNS = range(1,5)

make_images=True
make_video_db=False

# def get_vid_length(input_video):
#     print(input_video)
#     result = subprocess.Popen('ffprobe -i input_video -show_entries format=duration -v quiet -of csv="p=0"', stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#     output = result.communicate()
#     return output[0]

def process_video(input_video, i):

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)      # Opencv2 version 2 used "CV_CAP_PROP_FPS"
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(n_frames)
    if make_images:
        success,image = cap.read()
        frame = 0
        id = 0
        while success:
            if frame % IMAGE_SAMP == 0:
                cv2.imwrite(IMG_DIR + 'part_{0}/frame_{1}_img_{2}.jpg'.format(i, frame, id), image)     # save frame as JPEG file
                id += 1
            success,image = cap.read()
            frame += 1
    return n_frames

with open(CSV_FILE, mode="w") as f:
    # write header
    f.write('org_video,label,start_frm,video_id\n')
    id = 0

    for i in RUNS:
        video = DATA_DIR + 'new_part{0}.mp4'.format(i)
        vid_length = process_video(video, i)
        if make_video_db:
            for striding in range(0, vid_length, CLIP_LEN):
                f.write('{0},0,{1},{2}\n'.format(video, striding, id))
                id += 1
