# Fast Multi Style Transfer
Implementation of Google Brain's [A Learned Representation For Artistic Style](https://arxiv.org/pdf/1610.07629v2.pdf) in Tensorflow.
You can mix various type of style image using just One Model and it's still Fast!

<p>
<img src="result/result.jpg" width="1000" height="550" />
</p>
Figure1. Using one model and making multi style transfer image. Center image is mixed with 4 style

This paper is next version of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
and [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).
These papers are fast and nice result, but one model make only one style image.


## Implementation Details
The key of this paper is Conditional instance normalization.

<p>
<img src="result/conditional_instance_norm.jpg" />
</p>

Instance normalization is similar with batch normalization,
but it doesn't accumulate mean(mu), variance(alpha).
Conditional instance normalization have N scale(gamma) and N shift(beta). N means style number.
This mean when you add new style, you just train new gamma and new beta.
See the below results.

From Scratch.
Train weight, bias, gamma, beta

<p>
<img src="result/style01_01.gif" />
</p>
(40000 iteration)

Fine-Tuned. Gradually change to new style
Train new gamma, beta.

<p>
<img src="result/style02_01.gif" />
<img src="result/style03_01.gif" />
</p>
(4000 iteration, 1/10 scratch)


## Usage
example command lines are below and train_style.sh, test_style.sh
### Train

From Scratch
> python main.py -f 1 -gn 0 -p MST -n 5 -b 16 -tsd images/test -scw 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -sti images/style_crop/0_udnie.jpg


Fine-Tuned
> python main.py -f 1 -gn 0 -p MST -n 1 -b 16 -tsd images/test -scw 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 -sti images/style_crop/1_la_muse.jpg 
> python main.py -f 1 -gn 0 -p MST -n 1 -b 16 -tsd images/test -scw 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 -sti images/style_crop/10_Yellow_sunset.jpg


### Test
Single style
> python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
> python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Multi Style
> python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0.5 0.5 0 0 0 0 0 0 0 0 0 0 0 0 0 0
> python main.py -f 0 -gn 0 -p MST -tsd images/test -scw 0.4 0.3 0.2 0.1 0 0 0 0 0 0 0 0 0 0 0 0



### Requirements
- TensorFlow 1.0.0
- Python 2.7.12, Pillow 3.4.2, scipy 0.18.1, numpy 1.11.2

### Attributions/Thanks
This project borrowed some code from [Lengstrom's fast-style-transfer.](https://github.com/lengstrom/fast-style-transfer)
And Google brain's code is [here](https://github.com/tensorflow/magenta) (need some install)
