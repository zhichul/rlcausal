import sys

file = sys.argv[1]

names = file.split("/") 
if names[0] == "checkpoints":
    names.pop(0)
if names[-1] == "huggingface":
    names.pop(-1)
if names[-1] == "actor":
    names.pop(-1)

print("/".join(names))