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
```
Loading parameters for the SortBot
> 1,2,3
[[1 2 3]] [[1 2 3 6]]
```


