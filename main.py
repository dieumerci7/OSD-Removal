import click
from osd import OSDRemover
from time import time


@click.command(help="")
@click.option("--video-path", type=str, help="testing video path")
@click.option("--video-output-path", type=str, 
    default='output.mp4', help="path to save resulting video")
def main(video_path, video_output_path):

    osd = OSDRemover()
    start_time = time()

    clean_video = osd.remove_OSD_quick(video_path)

    end_time = time()
    print(f'Elapsed time = {end_time - start_time}s')
    osd.write_video(clean_video, video_output_path)


if __name__ == '__main__':
    main()