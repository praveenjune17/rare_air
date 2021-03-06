# rare_air
Transformer model for Text summarisation and machine translation created using TF2.

# Features
  a) Pointer generator can be enabled
  b) BERT SCORE Evaluation metric
  c) Visualisation script to create summary statistics of the datasets
  d) Input can be fed as tfds or in csvs
  e) Tensorboard attached
  f) prediction dynamics for 1st batch can be written out to a file
  g) Early stop using the monitored metrics like BERT SCORE, ROUGE SCORE 
  h) Beam search enabled inference
  i) Label smoothing 
  j) Optimized with keras mixed precision policy
  h) Automatic vocab file generation
  
# Features to be added 
  *) Add requirements
  *) Housekeeping
  *) Generalize to NMT
  *) New transformer.py to replace existing encoders and decoders with hugging face transformers 
  *) Visualize the attention weights
  *) Replace tf.pyfunction to tf.function
  
# Ideas adapted from 
  a) https://arxiv.org/pdf/1902.09243v2.pdf
  b) http://karpathy.github.io/2019/04/25/recipe/
  c) https://github.com/policeme/transformer-pointer-generator
  d) https://github.com/raufer/bert-summarization
  
  
