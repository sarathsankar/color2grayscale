# color2grayscale
CNN based model to convert any RGB image to Gray-scale

### Environment
- Ubuntu 18.04
- Python 3.6.*
- Tensorflow 1.11.0

### Directories
- `model/` keep saved model files.
- `images/intermediate/` any intermediate files/images.
- `images/predicted/` predicted images or output gray-scale images from trained model file.
- `images/source/` RGB source images.
- `images/target/` target gray-scale images of source color images for training.
- `images/validate/` fresh color images for validation, not a subset of source images.

Use [color2gray](https://github.com/sarathsankar/Image-Preprocessing/blob/master/color2gray.py) to genarate gray-scale images from source images for training. Configure directory paths and parameters in `config.ini` file.

### How to use

```python
from color2gray import COLOR2GRAY as C2G
from predict import PREDICT as PRED
## Train ##

# Adopt type1 or type2 for training
# Type 1, w.r.t to config file

c2g_obj = C2G()
c2g_obj.read_dataset()
c2g_obj.train()

# Type 2
read_args = {
    'color_img_dir': color_image_path,
    'target_gray_img_dir': target_grayscale_image_path,
    'model_dir': model_directory_path
}
c2g_obj = C2G(**read_args)
c2g_obj.read_dataset(img_type='png')
train_args = {
    'batch_size': 32,
    'epoch_num': 50
}
c2g_obj.train(**train_args)

## Predict
# Type 1
pred_obj = PRED()
pred_obj.predict()

# Type 2
pred_obj = PRED()
pred_args = {
    'color_img_dir': testing_input_image_directory_path,
    'predict_img_dir': result_image_directory_path,
    'img_type': png/jpg
}
pred_obj.predict(**pred_args)
```