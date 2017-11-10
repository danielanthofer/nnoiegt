# nnoiegt

# Neural Network for Open Information Extraction from German Text

Master's thesis

Abstract
--------
Systems that extract information from natural language texts usually need
to consider language-dependent aspects like vocabulary and grammar.
Compared to the development of individual systems for different languages,
development of multilingual information extraction (IE) systems has the
potential to reduce cost and effort. One path towards IE from different
languages is to port an IE system from one language to another. PropsDE is
an open IE (OIE) system that has been ported from the English system PropS
to the German language. Here we perform an analysis and a comparison of
PropS and PropsDE. Then we present a deep-learning based OIE system
for German, which mimics the behaviour of PropsDE. The precision in the
imitation of PropsDE is 24.4 %. Our model produces many extractions that
appear promising, but are not fully correct.

Requirements
------------
To start *extract.py*, you need working installations of
- PropsDE (https://github.com/UKPLab/props-de)
- Mate Tools (http://www.ims.uni-stuttgart.de/forschung/ressourcen/werkzeuge/matetools.html)
- JobImText (https://www.lt.informatik.tu-darmstadt.de/de/software/jobimtext/)
in the *external* folder

Apart from that, you need
- Python 2.7
- TensorFlow 1.3
- Scikit-learn
- numpy

The model can be found at https://www.dropbox.com/s/2sw8h06vs1vz541/train_3x1024_GRU_bidirectional_distinct_emb128d.zip?dl=0

Run the extractor
-----------------
```
  python extract.py example.txt
```

Remarks
-------
Train the model (adapt the config file accordingly):
```
	python train.py train configs/train_3x1024_GRU_bidirectional_distinct_emb128d.ini
```

Generate embeddings and training data with IPython notebooks (adapt the settings in these notebooks according to your needs):
1. *generateTrainingData*
2. *generateEmbeddings* 
3. *formatEmbeddingsAndTrainingData* 

To recreate word precision evaluation result, start training and wait for output "test word precision".

*Note: train.py is much faster if you run it directly on your system with individually compiled TensorFlow*
