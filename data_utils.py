import numpy as np

class Data_Utils():
	def __init__(self, file, batch_size, num_steps):
		self.file = file
		self.batch_size = batch_size
		self.num_steps = num_steps
		self.counter = 0

		self.process_data()
		self.create_batches()

	def process_data(self):
		with open(self.file,'r') as f:
			raw_data = f.read()
		self.unique_notes = list(set(raw_data))
		self.notes_to_ind = dict()
		for i in range(len(self.unique_notes)):
			self.notes_to_ind[self.unique_notes[i]] = i
		self.ind_to_notes = dict(zip(self.notes_to_ind.values(), self.notes_to_ind.keys()))
		self.data = [self.notes_to_ind[c] for c in raw_data]

	def create_batches(self):
		self.num_batches = len(self.data)//(self.batch_size * self.num_steps)
		self.counter = 0
		if self.num_batches == 0:
			raise Exception('Not enough data provided. Reduce batch size or number of recurrent steps')

		xdata = np.array(self.data)
		ydata = np.copy(xdata)
		ydata[:-1] = xdata[1:]
		ydata[-1] = self.notes_to_ind['\n']

		self.x_batches = []
		self.y_batches = []

		for n in range(self.num_batches):
			xbatch = np.zeros(shape=(self.batch_size, self.num_steps))
			ybatch = np.zeros(shape=(self.batch_size, self.num_steps))
			c = 0
			for b in range(self.batch_size):
				xbatch[b] = xdata[c:c+self.num_steps]
				ybatch[b] = ydata[c:c+self.num_steps]
				c+=self.num_steps
			self.x_batches.append(xbatch)
			self.y_batches.append(ybatch)

	def next_batch(self):
		x, y = self.x_batches[self.counter], self.y_batches[self.counter]
		self.counter += 1
		return x, y

	def reset(self):
		self.counter = 0

	def nums_to_notes(self, arr):
		return [self.ind_to_notes[c] for c in arr]

	def notes_to_nums(self, arr):
		return [self.notes_to_ind[c] for c in arr]
