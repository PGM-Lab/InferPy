from functools import reduce

class replicate():

	sizes = [1]

	def __init__(self,size):
		replicate.sizes.append(size)
	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		replicate.sizes.pop()

	@staticmethod
	def getSize():
		if len(replicate.sizes) == 0:
			return None
		return reduce(lambda x, y: x * y, replicate.sizes)

	@staticmethod
	def printSize():
		print("size is "+str(replicate.getSize()))

	@staticmethod
	def inReplicate():
		return len(replicate.sizes)>1

if __name__ == "__main__":

	import inferpy as inf

	with inf.replicate(size=10):
		print("some code...")
		print(inf.replicate.getSize())

		with inf.replicate(size=2):
			print("some code...")
			print(inf.replicate.getSize())

		print(inf.replicate.getSize())

	with inf.replicate(size=50):
		print("some code...")
		print(inf.replicate.getSize())

	print(inf.replicate.getSize())
