# RNN Language Processing
Language Processing with Recurrent Neural Network written using only numpy.

The program splits the sentences from kafka.txt file and then tokenizes each word. For training, most probable words are used. Before training, each sentence is added with a SENTENCE_START_TOKEN and SENTENCE_END_TOKEN to mark the beginning and end of each sentence and an UNKNOWN_TOKEN for words which are not very often used. Finally the RNN architecture learns the sentence formation and generates a sentence with some size limit.

## Generated text:
Gregor had very circumstances protruded bleeding chance wishing hoped breathe gentle on broom career conversation frame wondered bed women interest peacefully frenzy frequently deranged them men remarkable start usually angle happened until Sunday