Pytorch implementation of CNN and GAN based super-resolution models.

You need to download CelebA data to the root directory and build the dataset if you want to train from the beginning.

To train the model, you need to run the train.py in either directory cnn or directory gan. 

python train.py --data_dir <../YOUR_TRAIN_IMAGE_DIRECTORY> --model_dir experiments/<YOUR_MODEL_NAME> --model <YOUR_MODEL_NAME> --cuda <YOUR_CUDA> --optim <YOUR_OPTIMIZER> (--restore_file 'best' "IF YOU WANT TO TRAIN WITH WEIGHTS")

For example, if you want to train with data in your /data/train directory with the model "drrn_b1u9_model" with cuda0 and adam optimizer,

python train.py --data_dir ../data/train --model_dir experiments/drrn_b1u9_model --model drrn_b1u9 --cuda cuda0 --optim adam


To test our trained model,
python evaluate.py --data_dir <../YOUR_TEST_IMAGE_DIRECTORY> --model_dir experiments/<YOUR_MODEL_NAME> --model <YOUR_MODEL_NAME> --cuda <YOUR_CUDA> 


python evaluate.py --data_dir ../data/test --model_dir experiments/drrn_b1u9_model --model drrn_b1u9 --cuda cuda0