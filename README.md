# DIP_final


## Enviroments

the code is run on Arch Linux

### Neural Style Transfer
Follow the instruction in https://github.com/crowsonkb/style-transfer-pytorch

### SANet
please download the three pretained model following the instruction in https://github.com/GlebSBrykin/SANET

### MCCNet
please download the three pretained model and put it in experiments/ directory following the instruction in https://github.com/diyiiyiii/MCCNet

### Linear Style Transfer
please download models.zip and unzip it in the LinearStyleTransfer directory following the instruction in https://github.com/sunshineatnoon/LinearStyleTransfer


### Python Packages
```
pip install -r requirements.txt
```


## How to run

* For Image Style Transfer
  1. put all content images in a folder, default is content/
  2. put all style images in a folder, default is style/
  3. write a json file to specify customize instruction of each content-style pair, run default sytle transform is no config file
  4. run 
```
python image_style_transfer.py --content_dir <YOUR_CONTENT_DIR> --style_dir <YOUR_STYLE_DIR> --output_dir <YOUR_OUTPUT_DIR> (optional)--cmd_config <CONFIG> (optional)--nst_algo <NST_ALGO>
```

* For video Style Transfer
run
```
python video_style_transfer.py --video <PATH_TO_VIDEO> --style <PATH_TO_STYLE_IMAGE> --output_dir <PATH_TO_OUTPUT_DIR> (optional)--resize <new_H> <new_W> (optional)--rotate <ANGLE> (optional)--preserve_color (optional)--add_noise (optional)--nst_algo <NST_ALGO>
```








## citation


#### For SANet
No citation info, but the original implementation is https://github.com/GlebSBrykin/SANET

#### For LinearStyleTransfer Implementation
```
@inproceedings{li2018learning,
    author = {Li, Xueting and Liu, Sifei and Kautz, Jan and Yang, Ming-Hsuan},
    title = {Learning Linear Transformations for Fast Arbitrary Style Transfer},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2019}
}
```

#### For MCCNet
```
@inproceedings{deng:2020:arbitrary,
  title={Arbitrary Video Style Transfer via Multi-Channel Correlation},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Huang, haibin and Ma chongyang and Xu, Changsheng},
  booktitle={AAAI},
  year={2021},
 
}
```