# RNN Language Processing
Language Processing with Recurrent Neural Network written using only numpy.

The program splits the sentences from kafka.txt file and then tokenizes each word. For training, most probable words are used. Before training, each sentence is added with a SENTENCE_START_TOKEN and SENTENCE_END_TOKEN to mark the beginning and end of each sentence and an UNKNOWN_TOKEN for words which are not very often used. Finally the RNN architecture learns the sentence formation and generates a sentence with some size limit.

## Generated text:
His previous resolved noticing o'clock upwards subordinates convinced occupational kitchen towards draught instrument reported 's desk , seen lungs minutes picked On agreed so their of shoulder be unhappy accept peering to me brother strong also out ; for against endless like packed roll object Gregor joined spare he directions with other tone shoulders tightly body itch expectations diligently alert nearby seen mingle especially try The noticed anything noise disturbed ever but overcome noticeable cash sending mother carefully voice side
