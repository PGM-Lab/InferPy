import inferpy as inf


with inf.replicate(size=50,name="A"):
    # Define some random variables here
    print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")


with inf.replicate(size=10):
    # Define some random variables here
    print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")

    with inf.replicate(size=2, name="C"):
        # Define some random variables here
        print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")
        print(inf.replicate.in_replicate())

# Define some random variables here
print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")


### existing replicate construct can be reused.
### This is done by ommiting the size argument and only setting the name with the name of
### an existing one.


with inf.replicate(name="A"):
    with inf.replicate(name="C"):
        # Define some random variables here
        print("Variable replicated " + str(inf.replicate.get_total_size()) + " times")



inf.replicate.get_active_replicate()