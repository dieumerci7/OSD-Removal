import cv2
import click


def read_video(path:str) -> list:
    video_cap = cv2.VideoCapture(path)

    if not video_cap.isOpened():
        raise ValueError("Could not open video file")

    frames = list()
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break 
        frames.append(frame)
    return frames


@click.command(help="")
@click.option("--video-path", type=str, help="testing video path")
def main(video_path):

    video = read_video(video_path)



if __name__ == '__main__':
    main()