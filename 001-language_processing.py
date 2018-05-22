print("Language Processing! v01")
print("Loading libraries...")
import numpy as np
import nltk
import itertools
import operator
from datetime import datetime
import sys

intented_vocabulary_size = 2000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print("Reading Book...")

data = open('kafka.txt', 'r').read()
sentences = [sent + "." for sent in data.split(". ")]
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]

# print(sentences[1])
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

print("Found %d unique words tokens." % len(word_freq.items()))

vocab = word_freq.most_common(intented_vocabulary_size)

index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)

word_to_index = dict([(w,i) for i, w in enumerate(index_to_word)])

print("Intented vocabulary size of %d" % intented_vocabulary_size)
for i, sent in enumerate(tokenized_sentences):
	tokenized_sentences[i] = [w if w in index_to_word else unknown_token for w in sent]

print("\n Example sentence: %s" % sentences[0])
print("\n Example sentence after processing: %s" % tokenized_sentences[0])

X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

def softmax(z):
		return np.exp(z) / np.sum(np.exp(z))

class RNN:
	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
		# Assign instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		# Random initialization of network parameters
		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

	def forward_propagation(self, x):
		# The total number of time steps
		T = len(x)
		#During forward porpagation, we save all the hidden states in s since we need them
		s = np.zeros((T+1, self.hidden_dim)) # additional element is to set 0
		s[-1] = np.zeros(self.hidden_dim)
		# outputs are also saved for each time
		o = np.zeros((T, self.word_dim))
		# for each time step
		for t in np.arange(T):
			s[t] = np.tanh(self.U[:,x[t]] + self.W.dot(s[t-1]))
			o[t] = softmax(self.V.dot(s[t]))
		return [o, s]

	def predict(self, x):
		o, s = self.forward_propagation(x)
		return np.argmax(o, axis=1)

	def calculate_total_loss(self, x, y):
		L = 0
		# for each instance
		for i in np.arange(len(y)):
			o, s = self.forward_propagation(x[i])
			correct_word_predictions = o[np.arange(len(y[i])), y[i]]
			# add to the loss based on how off we are
			L += -1*np.sum(np.log(correct_word_predictions))
		return L

	def calculate_loss(self, x, y):
		N = np.sum((len(y_i) for y_i in y))
		return self.calculate_total_loss(x, y)/N

	def bptt(self, x, y):
		T = len(y)
		# performing forward propagation
		o, s = self.forward_propagation(x)
		# gradient initialisation
		dLdU = np.zeros(self.U.shape)
		dLdV = np.zeros(self.V.shape)
		dLdW = np.zeros(self.W.shape)

		delta_o = o
		delta_o[np.arange(len(y)), y] -= 1.
		for t in np.arange(T)[::-1]:
			dLdV += np.outer(delta_o[t], s[t].T)
			delta_t = self.V.T.dot(delta_o[t])*(1-s[t]**2)
			for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
				dLdW += np.outer(delta_t, s[bptt_step-1])
				dLdU[:, x[bptt_step]] += delta_t
				delta_t = self.W.T.dot(delta_t)*(1-s[bptt_step-1]**2)
		return [dLdU, dLdV, dLdW]

	def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
		# Calculate the gradients using backpropagation.
		bptt_gradient = self.bptt(x, y)
		model_parameters = ['U', 'V', 'W']
		# gradient check
		for pidx, pname in enumerate(model_parameters):
			parameter = operator.attrgetter(pname)(self)
			print("Perfrming gradient check for parameters %s with size %d." % (pname, np.prod(parameter.shape)))
			# iterate over each element of the parameter matrix
			it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
			while not it.finished:
				ix = it.multi_index
				original_value = parameter[ix]
				parameter[ix] = original_value + h
				gradplus = self.calculate_total_loss([x], [y])
				parameter[ix] = original_value - h
				gradminus = self.calculate_total_loss([x], [y])
				estimated_gradient = (gradplus - gradminus)/(2*h)
				parameter[ix] = original_value
				# The gradient for this parameter calculated using backpropagation
				backprop_gradient = bptt_gradient[pidx][ix]
	            # calculate The relative error: (|x - y|/(|x| + |y|))
				relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
	            # If the error is to large fail the gradient check
				if relative_error > error_threshold:
					print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
					print("+h Loss: %f" % gradplus)
					print("-h Loss: %f" % gradminus)
					print("Estimated_gradient: %f" % estimated_gradient)
					print("Backpropagation gradient: %f" % backprop_gradient)
					print("Relative Error: %f" % relative_error)
					return
				it.iternext()
			print("Gradient check for parameter %s passed." % (pname))

	def sgd_step(self, x, y, learning_rate):
		# calculating gradients
		dLdU, dLdV, dLdW = self.bptt(x, y)
		# change parameters according to gradient
		self.U -= learning_rate*dLdU
		self.V -= learning_rate*dLdV
		self.W -= learning_rate*dLdW

	def train_with_sgd(self, x, y, learning_rate=0.05, nepoch=1000, evaluate_loss_after=5):
		losses = []
		num_examples_seen = 0
		for epoch in range(nepoch):
			# optionally evaluate the loss
			if (epoch % evaluate_loss_after == 0):
				loss = self.calculate_loss(x, y)
				losses.append((num_examples_seen, loss))
				time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				print("%s: Loss after number of samples seen = %d, epoch = %d: %f" % (time, num_examples_seen, epoch+1, loss))
				# adjusting learning rate if loss increases
				if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
					learning_rate = learning_rate * 0.5
					print("Setting learning rate to %f" % learning_rate)
				sys.stdout.flush()
			# for each training example
			for i in range(len(y)):
				# 1 SGD per step
				model.sgd_step(x[i], y[i], learning_rate)
				num_examples_seen += 1

	def generate_sentence_prob(self):
		new_sentence = [word_to_index[sentence_start_token]]
		while not new_sentence[-1] == word_to_index[sentence_end_token]:
			next_word_probs = self.forward_propagation(new_sentence)[0]
			sampled_word = word_to_index[unknown_token]
			while sampled_word == word_to_index[unknown_token]:
				samples = np.random.multinomial(1, next_word_probs[-1])
				sampled_word = np.argmax(samples)
			new_sentence.append(sampled_word)
		sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
		return sentence_str

	def generate_sentence(self, num_sentences=1, sentence_min_length=7):	
		for i in range(num_sentences):
			sent = []
			while len(sent) < sentence_min_length:
				sent = self.generate_sentence_prob()
			print(" ".join(sent))


vocabulary_size = min(intented_vocabulary_size, len(index_to_word))

np.random.seed(0)
model = RNN(vocabulary_size)
losses = model.train_with_sgd(X_train[:100], Y_train[:100], nepoch=50, evaluate_loss_after=10)

model.generate_sentence()
