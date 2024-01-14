from args import args
from models.graphconv import GraphST

if __name__ == "__main__":
    GraphST(args=args, build=True)
    print()
