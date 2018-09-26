import os

folder = "./modules"
exclude_sections = ["inferpy\.models\.params module"+"\n", "Module contents"+"\n"]

for root, dirs, files in os.walk(folder):
    for fname in files:

        with open(folder+"/"+fname) as f:
            content = f.readlines()


        with open(folder+"/"+fname, "r") as f:
            lines = f.readlines()

        delflag = False
        with open(folder+"/"+fname,"w") as f:
            for line in lines:
                if line in exclude_sections:
                    delflag = True
                elif line.startswith("inferpy\."):
                    delflag = False

                if not delflag:
                    f.write(line)

