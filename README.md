# rare_air
seq2seq  train and infer skeleton for Text summarisation and machine translation

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
  
# Features to be added (hopefully with help from others)
  *) Add requirements
  
  a) Gradient accumulation
  b) Replace existing encoders and decoders with hugging face transformers 
  c) Visualizing the attention weights
  d) Replace tf.pyfunction to tf.function
  e) Add BEAM search as part of training 
  
# Inspired from 
  a) https://arxiv.org/pdf/1902.09243v2.pdf
  b) http://karpathy.github.io/2019/04/25/recipe/
  c) https://github.com/policeme/transformer-pointer-generator
  d) https://github.com/raufer/bert-summarization
  
  
