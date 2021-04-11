import pathlib
import tensorflow as tf

class SemisegDataLoader():
  """
  Keras - Tensorflow - Semiseg dataloader implementation
  
  tr_imgs_path    - Training imgs directory path
  tr_masks_path   - Training masks directory path
  te_imgs_path    - Testing imgs directory path
  te_masks_path   - Testing masks directory path
  """
  def __init__(self, tr_imgs_path, tr_masks_path, te_imgs_path, te_masks_path):
    self.tr_imgs_path = tr_imgs_path
    self.tr_masks_path = tr_masks_path
    self.te_imgs_path = te_imgs_path
    self.te_masks_path = te_masks_path
  

  def create_dataset(self, resize_img=(128, 128)):
    # Datadirs
    tr_imgs_datadir = pathlib.Path(self.tr_imgs_path)
    tr_masks_datadir = pathlib.Path(self.tr_masks_path)
    
    te_imgs_datadir = pathlib.Path(self.te_imgs_path)
    te_masks_datadir = pathlib.Path(self.te_masks_path)

    # # Lists of file names from datadirs
    tr_imgs_paths = list(tr_imgs_datadir.glob('*.jpg'))
    tr_masks_paths = list(tr_masks_datadir.glob('*.png'))

    te_imgs_paths = list(te_imgs_datadir.glob('*.jpg'))
    te_masks_paths = list(te_masks_datadir.glob('*.png'))

	# Decode & Normalize & Resize
    def path_to_tensor_img(img_path, is_tr=True):
      if is_tr:
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)
        image = tf.image.resize(image, resize_img)
        image = tf.cast(image, tf.float32) / 255.0
        return image
      else:
        mask = tf.io.read_file(img_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, resize_img)
        mask = mask // 255 # normalize
        return mask

    tr_images = list(map(lambda p: path_to_tensor_img(p, True), sorted(map(lambda x: str(x), tr_imgs_paths))))
    tr_segmentation_masks = list(map(lambda p: path_to_tensor_img(p, False), sorted(map(lambda x: str(x), tr_masks_paths))))


    te_images = list(map(lambda p: path_to_tensor_img(p, True), sorted(map(lambda x: str(x), te_imgs_paths))))
    te_segmentation_masks = list(map(lambda p: path_to_tensor_img(p, False), sorted(map(lambda x: str(x), te_masks_paths))))

    # tuples
    train_ds_tuples = tf.data.Dataset.from_tensor_slices((tr_images, tr_segmentation_masks ))
    test_ds_tuples = tf.data.Dataset.from_tensor_slices((te_images, te_segmentation_masks ))

    return { 'train': train_ds_tuples, 'test': test_ds_tuples }
