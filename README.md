# Draw What You Hear: High-fidelity Image Generation and Manipulation via SoundAdapter


## Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda env create -f env.yml
```
## Inference
Download the pretrained [checkpoint](https://huggingface.co/YSYS1103/SoundAdapter)
- Sound-oriented Image Generation
```Shell
python ./optimization/SoundAdapter_SD.py
```
- Sound-guided Image Manipulation
```Shell
python ./optimization/SoundAdapter_GAN.py
```
- MultiModelMix
```Shell
python ./optimization/SoundMix.py
``` 
- SoundMix
```Shell
python ./optimization/SoundAddSoundMix.py
```
## References
@article{w2025soundadapter,
  title={Draw What You Hear: High-fidelity Image Generation and Manipulation via SoundAdapter},
  author={Mingjie Wang, Song Yuan, Xianfeng Han and Yizi Li},
  journal={IEEE transactions on neural networks and learning systems},
  year={2025},
}
