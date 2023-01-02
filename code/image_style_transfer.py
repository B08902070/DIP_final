import os
import cv2
import json
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from Reversible_Transform.Reversible_Transform_Console import Reversible_Transform_Console
from Add_noise.add_noise import add_noise

from Preserve_Color import Preserve_Color_Luminance

def make_temp_dir(dir_name, img, img_name):
    if os.path.exists(dir_name):
        subprocess.run(['rm', '-rf', dir_name], shell=True)
    os.makedirs(dir_name)
    cv2.imwrite(f"{dir_name}/{img_name}.jpg", img)

def run_style_transfer_algo(algo, content_img, style_img, content_name, style_name, output_dir):
    if algo == 'LinearStyleTransfer':
        tmp_content_dir = "tmp_content"
        tmp_style_dir = "tmp_style"
        make_temp_dir(tmp_content_dir, content_img, content_name)
        make_temp_dir(tmp_style_dir, style_img, style_name)
        
        subprocess.run(['python', 'LinearStyleTransfer/TestArtistic.py', f'--contentPath {tmp_content_dir}', 
                        f'--stylePath {tmp_style_dir}', f'--outf {str(output_dir)}/{content_name}_{style_name}.jpg'])

    elif algo == "Gatys":
        subprocess.run(['Gatys/style_transfer', f'{tmp_content_dir}', 
                        f'{tmp_style_dir}', f'-o {str(output_dir)}/{content_name}_{style_name}.jpg'])

    elif algo == "MCCNet":
        tmp_content_path = f'{content_name}.jpg'
        tmp_style_path = f'{style_name}.jpg'

        subprocess.run(['python', 'MCCNet/test_video.py', f'--content {tmp_content_path}', 
                        f'--style {tmp_style_path}', f'--output {str(output_dir)}'])

    elif algo == "SANet":
        tmp_content_path = f'{content_name}.jpg'
        tmp_style_path = f'{style_name}.jpg'

        subprocess.run(['python', 'SANet/Eval.py', f'--content {tmp_content_path}',
                        f'--style {tmp_style_path}', f'--output {str(output_dir)}'])
        



def get_images_and_names(image_dir):
    names = []
    imgs = []
    for p in image_dir.glob('*'):
        dir_sep = '\\' if '\\' in str(p) else '/'
        name = str(p).split(dir_sep)[-1].split('.')[0]
        names.append(name)

        img = cv2.imread(p)
        imgs.append(img)
    return names, imgs

def image_style_transfer(args):
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    config_dict = json.load(args.cmd_config)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    content_names, content_imgs = get_images_and_names(content_dir)
    style_names, style_imgs = get_images_and_names(style_dir)

    for ci in range(len(content_imgs)):
        for si in range(len(style_imgs)):
            """ if content-style pair not in command, then do nothing before and after style transfer """
            if content_names[ci] not in config_dict.keys():
                run_style_transfer_algo(args.nst_algo, content_img=content_imgs[ci], style_img=style_imgs[si],
                                        content_name=content_names[ci], style_name=style_names[si], output_dir=output_dir)
            content_dict = config_dict[content_names[ci]]
            if style_names[si] not in content_dict.keys():
                run_style_transfer_algo(args.nst_algo, content_img=content_imgs[ci], style_img=style_imgs[si],
                                        content_name=content_names[ci], style_name=style_names[si], output_dir=output_dir)

            
            cmd_dict = content_dict[style_names[si]]
            """ do reversible transform foreward if mentioned in config """
            RT_console = Reversible_Transform_Console()
            for RT_dict in cmd_dict['Reversible Transform']:
                fn_name = RT_dict['fn_name']
                kwargs = RT_dict['kwargs']
                RT_console.load_transform_ops(fn_name, kwargs)

            """add noise"""
            content_img = content_imgs[ci]
            if cmd_dict['add_noise']:
                content_img = add_noise(content_img)

            """Reversible Transform forward"""
            content_img = RT_console.forward(content_imgs[ci])
            
            """do image style transfer"""
            run_style_transfer_algo(args.nst_algo, content_img=content_imgs[ci], style_img=style_imgs[si],
                                    content_name=content_names[ci], style_name=style_names[si], output_dir=output_dir)
            
            stylized_path = list(output_dir.glob(f'{content_names[ci]}_{style_names[si]}*'))[0]
            stylized_img = cv2.imread(stylized_path)

            """ do reversible transform backward"""
            stylized_img = RT_console.backward(stylized_img)

            """ do preserve color transfer if set True in config file"""
            if cmd_dict['preserve_color'] == True:
                out_img = Preserve_Color_Luminance.luminance_only_transfer(content_img=content_img, stylized_img=stylized_img)
            
            cv2.imwrite(stylized_path, out_img)
            

                


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--content_dir", help="path to content dir", type=str, default='../data/content/')
    parser.add_argument("--style_dir", help="path to style dir", type=str, default='../data/style/')
    parser.add_argument("--output_dir", help="path to output dir of stylized image", type=str, default='../output/')
    parser.add_argument("--cmd_config", help="path to the json file that contain the cmd config of content-style pair",
                         type=str, default=None)
    parser.add_argument("--nst_algo", help='choose the transfer method among [Gatys, SANet, LinearStyleTransfer, MCCNet]', type=str, default='LinearStyleTransfer')

    args = parser.parse_args()

    image_style_transfer(args)