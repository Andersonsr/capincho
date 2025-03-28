## Capincho ##

Image captioning composed of 3 modules: 1) a decoder only language model (OPT) for generating text, 2) a vision-language model CLIP for aligned representation of images and texts, 3) a embeddings mapper 
that maps CLIP embeddings to k OPT word embeddings.  

![captioning model pipeline](figs/mapper.png)

Some examples from coco dataset, after training for 2 epochs only while learning a prefix of length 10 (k=10):

![image 1](figs/captions_000000139684.jpg)

![image 2](figs/captions_000000246308.jpg)

![image 3](figs/captions_000000391722.jpg)

### Installation ###
````angular2html
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
````


### Usage ###
check the following files:

**extractFeatures.py** to extract the features vectors from coco dataset using CLIP or open CLIP.

**trainDecoder.py**  to train the mapper module and finetune OPT, or a OPT LoRA model.

**evaluateCaptioning.py** to qualitative evaluate results.

