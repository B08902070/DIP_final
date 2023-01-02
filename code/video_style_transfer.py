import os
import cv2
import subprocess
from pathlib import Path
from argparse import ArgumentParser

from Add_noise.add_noise import add_noise
from Reversible_Transform.Reversible_Transform_Console import Reversible_Transform_Console
from Preserve_Color import Preserve_Color_Luminance

def get_name(path):
    dir_sep = '\\' if '\\' in str(path) else '/'
    name = str(path).split(dir_sep)[-1].split('.')[0]
    return name

def video_to_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames=[]
    while True:
        ret, frame = cap.read()
        if ret is False:
            break
        frames.append(frame)
    cap.release()
    return frames

def frames_to_video(frames, video_path):
    frameSize = (frames[0].shape[1], frames[0].shape[0])
    out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

    for frame in frames:
        out.write(frame)

    out.release()

def run_style_transfer_algo(algo, video_path, style_path, output_dir):
    if algo == 'LinearStyleTransfer':
        video_name = get_name(video_path)
        style_name= get_name(style_path)
        subprocess.run(['python', 'LinearStyleTransfer/TestVideo.py', f'--videoPath {str(video_path)}', 
                        f'--stylePath {str(style_path)}', f'--outf {str(output_dir)}/{video_name}_{style_name}.mp4'])

    elif algo == "MCCNet":
        subprocess.run(['python', 'MCCNet/test_video.py', f'--content {str(video_path)}', 
                        f'--style {str(style_path)}', f'--output {str(output_dir)}'])




def video_style_transfer(args):
    video_path = Path(args.video)
    style_path = Path(args.style)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frames = video_to_frames(video_path)

    """add noise"""
    if args.add_noise:
        for i in range(len(frames)):
            frames[i] = add_noise(frames[i])

    RT_console = Reversible_Transform_Console()
    """ Reversible Resize"""
    if args.resize is not None:
        if len(args.resize) == 1:
            new_H, new_W = args.resize, args.resize
        elif len(args.resize) == 2:
            new_H, new_W = args.resize
        RT_console.load_transform_ops(transform_fn_name='Resize', transform_kwargs={"size1":new_H, "size2":new_W})

    """Reversible Rotate"""
    if args.rotate is not None:
        RT_console.load_transform_ops(transform_fn_name='Rotate', transform_kwargs={"angle":args.rotate})


    """Apply Reversible Transform forward"""
    for i in range(len(frames)):
        frames[i]=RT_console.forward(frames[i])

    
    """Video Style Transfer"""
    tmp_save_dir = Path('tmp_save_dir/')
    if tmp_save_dir.exists():
        subprocess.run(['rm', '-rf', str(tmp_save_dir)], shell=True)
    os.makedirs(tmp_save_dir)
    run_style_transfer_algo(algo=args.nst_algo, video_path=video_path, style_path=style_path, output_dir=tmp_save_dir)

    """Apply Reversible Transform backward"""
    video_path = list(tmp_save_dir.glob('*'))[0]
    ret_frames = video_to_frames(video_path=video_path)
    for i in range(len(frames)):
        ret_frames[i] = RT_console.backward(ret_frames[i])
    
    """Preserve Color"""
    if args.preserve_color:
        for i in range(len(frames)):
            ret_frames[i] = Preserve_Color_Luminance.luminance_only_transfer(content_img=frames[i], stylized_img=ret_frames[i])

    """write to video"""
    frames_to_video(frames=ret_frames, video_path=Path(args.output_dir))
            

                


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", help="path to video file", type=str)
    parser.add_argument("--style", help="path to style image", type=str)
    parser.add_argument("--output_dir", help="path to output dir of stylized video", type=str, default='output')
    parser.add_argument("--resize", help='specify the size if want to apply reversible resize', default=None, nargs="*")
    parser.add_argument('--rotate', help='specify the angle in degree if want to apply reversible rotate', default=None)
    parser.add_argument('--preserve_color', help='specify is want to apply preserve color transfer', action='store_true')
    parser.add_argument("--add_noise", help='specify if need to add noise to the video', action="store_true")
    parser.add_argument("--nst_algo", help='choose the transfer method among [LinearStyleTransfer, MCCNet]', type=str, deault='LinearStyleTransfer')

    args = parser.parse_args()

    video_style_transfer(args)