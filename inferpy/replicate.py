from functools import reduce # Valid in Python 2.6+, required in Python 3


class replicate():

	sizes = []

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
