## This readme is still in progress, please refer to the last section #to use our code for now, thank you!

# Super Resolution with CNNs and GANs

This is the code for our cs231n project.

**[Super Resolution with CNNs and GANs](https://github.com/yiyang7/cs231n_proj)**,
<br>
[Yiyang Li](https://github.com/yiyang7),
[Yilun Xu](https://github.com/Beehamer),
[Ji Yu](https://github.com/NaruSaku)
<br>

We investigated the problem of image super-resolution (SR), where we want to reconstruct high-resolution images from low-resolution images. We presented a residual learning framework to ease the training of the substantially deep network. Specifically, we reformulated the structure of the deep-recursive neural network to improve its performance. To further improve image qualities, we built a super-resolution generative adversarial network (SRGAN) framework, where we proposed several loss functions based on perceptual loss, i.e. SSIM loss and/ or total variation (TV) loss, to enhance the structural integrity of generative images. Moreover, a condition is injected to resolve the problem of partial information loss associated with GANs. 

The results show that our methods and trails can achieve equivalent performance on most of the benchmarks compared with the previous state-of-art methods, and out-perform them in terms of the structural similarity. Here are a few example outputs:

<img src='imgs/resultsfig.png'>

If you find this code useful in your research, please cite:

```
@inproceedings{densecap,
  title={Super Resolution with CNNs and GANs},
  author={Yiyang, Li and Yilun, Xu and Ji, Yu},
  year=2018,
}
```

## Installation
This project was implemented in [PyTorch 0.4](https://pytorch.org/#pip-install-pytorch) and [Python3](https://www.python.org/downloads/)

## Quickstart
1. __Build datasets__: First, you need to download the dataset. We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) as our dataset.

```
python build_dataset.py --data_dir ../img_align_celeba_test --output_dir ../data/cnn_faces --input_size 144 --output_size 144
```

2. __Train your experiment__

for cnn-based models, e.g SRCNN:
```
python train_cnn.py --data_dir ../data/cnn_faces --model_dir experiments/srcnn_model --model srcnn --cuda cuda0 --optim adam
```
for gan models, e.g SRGAN:
```
python train_gan.py --data_dir ../data/gan_faces --model_dir experiments/gan_model --model gan --cuda cuda0 --optim adam
```
If you want to train the model from last time, add this:
```
--restore_file "best"
```

3. __Perform hyperparameters search__

for cnn-based models, e.g SRCNN:
```
python search_hyperparams.py --data_dir ../data/cnn_faces --parent_dir experiments/learning_rate --model srcnn --model_type cnn
```
for gan models, e.g SRGAN:
```
python search_hyperparams.py --data_dir ../data/gan_faces --parent_dir experiments/learning_rate --model gan --model_type gan
```

4. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

5. __Evaluation on the test set__
```
python evaluate.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
## Guidelines for more advanced use
1. __save output of test images__

2. __save feature maps__


## To use our code:
Pytorch implementation of CNN and GAN based super-resolution models.

You need to download CelebA data to the root directory and build the dataset if you want to train from the beginning.

To train the model, you need to run the train.py in either directory cnn or directory gan. 

python train.py --data_dir <../YOUR_TRAIN_IMAGE_DIRECTORY> --model_dir experiments/<YOUR_MODEL_NAME> --model <YOUR_MODEL_NAME> --cuda <YOUR_CUDA> --optim <YOUR_OPTIMIZER> (--restore_file 'best' "IF YOU WANT TO TRAIN WITH WEIGHTS")

For example, if you want to train with data in your /data/train directory with the model "drrn_b1u9_model" with cuda0 and adam optimizer,

python train.py --data_dir ../data/train --model_dir experiments/drrn_b1u9_model --model drrn_b1u9 --cuda cuda0 --optim adam


To test our trained model,
python evaluate.py --data_dir <../YOUR_TEST_IMAGE_DIRECTORY> --model_dir experiments/<YOUR_MODEL_NAME> --model <YOUR_MODEL_NAME> --cuda <YOUR_CUDA> 


python evaluate.py --data_dir ../data/test --model_dir experiments/drrn_b1u9_model --model drrn_b1u9 --cuda cuda0


Our model uses preprocessed data. The original size of images in CelebA is 218 * 178. We use the code in matlab/preprocess.m to crop them to 144 * 144. Then you could downsize the picture to the size you want (e.g. 36 * 36 by scale_factor = 4 ) and then enlarge them to 144 * 144 by bicubic method.
