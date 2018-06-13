# Super Resolution with CNNs and GANs

This is the code for our cs231n project.

**[Super Resolution with CNNs and GANs](https://github.com/yiyang7/cs231n_proj)**,
<br>
[Yiyang Li](https://github.com/yiyang7),
[Yilun Xu](https://github.com/Beehamer),
[Ji Yu](https://github.com/NaruSaku),
<br>

We investigated the problem of image super-resolution (SR), reconstructing high-resolution images from low-resolution images, which has attracted attention because of its wide range of applications. In this paper, we mainly focus on improving super-resolution of human faces, and used CelebA as our dataset. Among all existing im- plementations, CNN-based models have enjoyed huge suc- cess. We presented a residual learning framework to ease the training of the substantially deep network. Specifically, we reformulated the structure of the deep-recursive neural network to improve its performance. To further solve the problem of recovering high frequency details of high res- olution images, we built a super-resolution generative ad- versarial network (SRGAN) framework, where we proposed several loss functions based on perceptual loss, i.e. SSIM loss and/ or total variation (TV) loss, to enhance the struc- tural integrity of generative images. Moreover, a condition is injected to resolve the problem of losing partial infor- mation associated GANs. The results show that our meth- ods and trails can achieve equivalent performance on most of the benchmarks compared with the previous state-of-art methods, and out-perform them in terms of the structural similarity.

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
