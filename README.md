# ESNAC: Embedding Space for Neural Architecture Compression

This is the PyTorch implementation of our paper:

Learnable Embedding Space for Efficient Neural Architecture Compression.<br/>Shengcao Cao\*, Xiaofang Wang\*, and Kris M. Kitani. ICLR 2019. \[[OpenReview](https://openreview.net/forum?id=S1xLN3C9YX)\] \[[arXiv](https://arxiv.org/abs/1902.00383)\].

## Requirements

We recommend you to use this repository with [Anaconda Python 3.7](https://www.anaconda.com/distribution/) and the following libraries:

- [PyTorch 1.0](https://pytorch.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [TensorFlow](https://www.tensorflow.org/) (Optional, only necessary if you would like to use TensorBoard to monitor the running of the job.)

## Usage

- Before running `compression.py`, you need to prepare the pretrained teacher models and put them at the folder `./models/pretrained`. You can choose to train them on your own with `train_model_teacher()` in `training.py`, or download them at:

  - [Google Drive](https://drive.google.com/open?id=1RgeUljIs5WeRuHYjWnWAZf_qkNa3O-IR)
  - [百度网盘 (BaiduYun)](https://pan.baidu.com/s/1p0_2YycHoau-wN5xw9xTuA) (Code: 9aru)

  We would like to point out that these provided pretrained teacher models are not trained on the full training set of CIFAR-10 or CIFAR-100. For both CIFAR-10 and CIFAR-100, we sample 5K images from the full training set as the validation set. The provided pretrained teacher models are trained on the remaining training images and are only used during the search process. The teacher accuracy reported in our paper refers to the accuracy of teacher models trained on the full training set of CIFAR-10 or CIFAR-100.

- Then run the main program:

  ```
  python compression.py [-h] [--network NETWORK] [--dataset DATASET]
                        [--suffix SUFFIX] [--device DEVICE]
  ```

  For example, run

  ```
  python compression.py --network resnet34 --dataset cifar100 --suffix 0 --device cuda
  ```

  and you will see how the ResNet-34 architecture is compressed on the CIFAR-100 dataset using your GPU. The results will be saved at `./save/resnet34_cifar100_0` and the TensorBoard log will be saved at `./runs/resnet34_cifar100_0`.

  Other hyper-parameters can be adjusted in `options.py`.

- The whole process includes two stages: searching for desired compressed architectures, and fully train them. `compression.py` will do them both. Optionally, you can use TensorBoard to monitor the process through the log files.

- After the compression, you can use the script `stat.py` to get the statistics of the compression results.

## Random Seed and Reproducibility

To ensure reproducibility, we provide the compression results on CIFAR-100 with random seed 127. This seed value is randomly picked. You can try other seed values or comment out the call of `seed_everything()` in `compression.py` to obtain different results. Here are the compression results on CIFAR-100 when fixing the seed value to 127:

| Teacher | Accuracy | #Params | Ratio | Times | f(x) |
| ---        | :---: | :---: | :---: | :---: | :---: |
| VGG-19     | 71.64% | 3.07M | 0.8470 |  6.54&times; | 0.9492 |
| ResNet-18  | 71.91% | 1.26M | 0.8876 |  8.90&times; | 0.9024 |
| ResNet-34  | 75.47% | 2.85M | 0.8664 |  7.48&times; | 0.9417 |
| ShuffleNet | 68.17% | 0.18M | 0.8298 |  5.88&times; | 0.9305 |

## Citation

If you find our work useful in your research, please consider citing our paper [Learnable Embedding Space for Efficient Neural Architecture Compression](https://openreview.net/forum?id=S1xLN3C9YX):

```
@inproceedings{
  cao2018learnable,
  title={Learnable Embedding Space for Efficient Neural Architecture Compression},
  author={Shengcao Cao and Xiaofang Wang and Kris M. Kitani},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=S1xLN3C9YX},
}
```

