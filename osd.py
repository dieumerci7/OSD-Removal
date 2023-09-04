import cv2
import numpy as np
from tqdm import tqdm
from typing import Union
from multiprocessing import Pool
from functools import partial


class OSDRemover:
    def __init__(
        self, 
        canny_t_lower:int = 20, 
        canny_t_upper:int = 150, 
        margin:float = 0.05,
        n_frames:Union[float, int] = 0.3,
        rotate_frames:bool = True
    ):
        
        # edge detection variables
        self.canny_t_lower = canny_t_lower
        self.canny_t_upper = canny_t_upper

        # margin for the center
        self.margin = margin

        # fraction of frames to calculate mean_frame
        self.n_frames = n_frames

        # whether to rotate selected frames for mean_frame, given centered OSD
        self.rotate_frames = rotate_frames


    def read_video(self, video_path:str) -> np.ndarray:
        ''' reading video frames given path '''
        video_cap = cv2.VideoCapture(video_path)

        if not video_cap.isOpened():
            raise ValueError("Could not open video file")

        frames = list()
        while True:
            ret, frame = video_cap.read()
            if not ret: break 
            frames.append(frame)
        
        frames = np.array(frames)
        return frames


    def write_video(self, video:list, video_save_path:str, fps:float = 30.0) -> None:
        ''' write given video to a file 
        
            Args:
               video (list): list of video frames of np.ndarray type
               video_save_path (str): file path where to save video
               fps (float): number of frames per second
        '''

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_height, frame_width, _ = video[0].shape

        # Create a VideoWriter object to save the video
        out = cv2.VideoWriter(
            video_save_path, fourcc, fps, (frame_width, frame_height))
        # Write each frame to the video file
        for frame in video:
            out.write(frame)
        out.release()


    def _mean_frame(self, video:np.ndarray) -> np.ndarray:
        ''' calculate mean frame of the video '''
        # selecting frames 
        assert self.n_frames < len(video)
        size = int(len(video) * self.n_frames) if isinstance(self.n_frames,
                                                             float) else self.n_frames
        frame_idx = np.random.choice(np.arange(video.shape[0]), size)
        subsample = video[frame_idx]

        # rotate frames 180 degrees if enabled
        if self.rotate_frames:
            rotated_subsample = np.rot90(subsample, k=2, axes=(1, 2)) 
            subsample = np.concatenate((subsample, rotated_subsample), axis=0)
        
        # calculate mean frame
        mean_frame = np.mean(
            subsample,
            axis=0).astype(np.uint8)
        return mean_frame


    def _augment_osd(self, edges:np.ndarray) -> np.ndarray:
        ''' osd mask augmentation with a rotate 90 degrees mask '''
        height, width = edges.shape
        x_center, y_center = height // 2, width // 2
        margin = int(max(height * self.margin, width * self.margin))
        
        # slice center of the mask
        center_edges = edges[
            x_center - margin: x_center + margin,
            y_center - margin: y_center + margin]

        # rotate 90 degrees center of mask: given OSD is located in center
        rotated_edges = np.rot90(center_edges)

        # update OSD mask 
        edges[
            x_center - margin: x_center + margin,
            y_center - margin: y_center + margin] = center_edges | rotated_edges

        return edges


    def _detect_osd_edges(self, frame:np.ndarray) -> np.ndarray:
        ''' calculating osd mask '''
        edges = cv2.Canny(frame, self.canny_t_lower, self.canny_t_upper, apertureSize=3)

        # morphological dilation to fill the edges
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours of the detected edges and fill them
        contours, _ = cv2.findContours(
            dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_image = np.zeros_like(edges)
        for contour in contours:
            cv2.fillPoly(filled_image, [contour], color=(255))
        
        # augmenting osd mask with a rotated 90 degrees mask
        filled_image = self._augment_osd(filled_image)
        return filled_image


    def _inpaint_OSD(self, frame:np.ndarray, mask:np.ndarray) -> np.ndarray:
        ''' remove osd on a given frame '''
        res = cv2.inpaint(frame, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA) 
        return res


    def remove_OSD_quick(self, video) -> list:
        ''' remove OSD from a given video (path of list of frames) '''
        if not isinstance(video, np.ndarray):
            video = self.read_video(video)

        # calculate mean frame on a sample of frames
        mean_frame = self._mean_frame(video)
        # calculate osd mask 
        osd_mask = self._detect_osd_edges(mean_frame)
        # remove osd
        clean_video = list()

        # with Pool(processes=4) as pool:
        #     args_list = [(frame, osd_mask) for frame in video]
        #     partial_inpaint = partial(self._inpaint_OSD, mask=osd_mask)
        #     clean_video = pool.map(partial_inpaint, args_list)

        with Pool(processes=4) as pool:
            args_list = [(self, frame, osd_mask) for frame in video]
            clean_video = pool.starmap(OSDRemover._inpaint_OSD, args_list)
            pool.close()
            pool.join()

        # for frame in tqdm(video):
        #     clean_frame = self._inpaint_OSD(frame, osd_mask)
        #     clean_video.append(clean_frame)
        return clean_video

    def remove_OSD(self, video) -> list:
        ''' remove OSD from a given video (path of list of frames) '''
        if not isinstance(video, np.ndarray):
            video = self.read_video(video)

        # calculate mean frame on a sample of frames
        mean_frame = self._mean_frame(video)
        # calculate osd mask
        osd_mask = self._detect_osd_edges(mean_frame)
        # remove osd
        clean_video = list()

        for frame in tqdm(video):
            clean_frame = self._inpaint_OSD(frame, osd_mask)
            clean_video.append(clean_frame)
        return clean_video
