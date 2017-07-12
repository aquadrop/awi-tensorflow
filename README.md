# awi-tensorflow
Attention With Intention or Hierarchical Attention Model in tensorflow

# references:
1. https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
2. https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py LINE 154 around
3. http://web.stanford.edu/class/cs20si/assignments/a3.pdf

# checklist
1. simple_seq2seq.py simulates plus op such as 2 + 3 = 5 using LSTM without embedding layer
2. simple_seq2seq2.py simulates plus op with embedding layer
3. simple_seq2seq3.py simulates sort op with embedding layer based on seq2seq model(vanilla)
4. simple_linear_model.py a testing file for tf.train.saver about saving and loading .meta file. And seperate the
    model in a defined class
   continue training with saved model
5. seq2seq_sort.py the complete model for sorting. It also supports continued training(or so called finetuing?)
    * input 3,2,1->1,2,3,6 6 is EOS symbol
6. att_seq2seq_sort.py self-developed attention model for sort training. I refereced
a. https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py
b. https://arxiv.org/pdf/1412.7449.pdf section 2.1
7. batch_transfter_plus_op.py demonstrates transferring hidden states across different batches training(==1)

   The task is about mod sum, but shifted by one batch.

   For instance:
   '''
   2,3 --> 0
   1,2 --> 0 (5 % 5=0)
   3,4--> 3
   2,2--> 2
   '''
8. HRED  Hierarchical Recurrent Encoder-Decoder network. Finally after about 30K iterations the model converges !!
'''
step and turn-1 328049 [[0 5]] [[ 6.  1.  0.]] 1.19209e-07 [array([1]), array([0]), array([6])] [[[ 0.  1.  0.  0.  0.  0.  0.]
  [ 1.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  1.]]]
step and turn-1 328059 [[3 3]] [[ 6.  3.  1.]] 7.6292e-05 [array([3]), array([1]), array([6])] [[[ 0.  0.  0.  1.  0.  0.  0.]
  [ 0.  1.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  1.]]]
step and turn-1 328069 [[2 3]] [[ 6.  0.  0.]] 3.82655e-05 [array([0]), array([0]), array([6])] [[[ 1.  0.  0.  0.  0.  0.  0.]
  [ 1.  0.  0.  0.  0.  0.  0.]
  [ 0.  0.  0.  0.  0.  0.  1.]]]
step and turn-1 328079 [[1 4]] [[ 6.  1.  0.]] 4.76836e-06 [array([1]), array([0]), array([6])] [[[ 0.  1.  0.  0.  0.  0.  0.]

Note that the print step is 10, therefore the second value does not repsond to the last printed one...
'''
9. awi_seq2seq develops attention-with-intention model. The task is to compute sort appended with sum%vol

   Task example:
   seq1: 2,1,3 --> 1,2,3,1 the last number is (2+1+3)%5
   seq2: 2,3,2 --> 2,2,3,3 the last number is (2+3+2+1) %5
   seq3: ...

   The model is "kind of " converged
```
Loading parameters for the SortBot(in which 6 is the EOS symbol
> 3,2,1
[[3 2 1]] [[1 2 3 6]]
```
Sample Output for att_seq2seq_sort.py
```
> 5,4,3,2,1,1,2,3,4,5
[[5 4 3 2 1 1 2 3 4 5]] [[1 1 2 2 3 3 4 4 5 5 6]]
```


