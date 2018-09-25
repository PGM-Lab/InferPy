fname = "./modules/inferpy.models.rst"



with open(fname) as f:
    content = f.readlines()



with open(fname, "r") as f:
    lines = f.readlines()

delflag = False
with open(fname,"w") as f:
    for line in lines:
        if line =="inferpy\.models\.params module"+"\n":
            delflag = True
        elif line.startswith("inferpy\."):
            delflag = False

        if not delflag:
            f.write(line)

