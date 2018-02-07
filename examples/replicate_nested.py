import inferpy as inf

with inf.replicate(size=50):
    # Define some random variables here
    print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")


with inf.replicate(size=10):
    # Define some random variables here
    print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")

    with inf.replicate(size=2):
        # Define some random variables here
        print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")


# Define some random variables here
print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")
