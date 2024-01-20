from args import args
from trainer import Trainer

if __name__ == "__main__":
    stxl_trainer = Trainer(training_args=args)
    stxl_trainer.train()
    # stxl_trainer.run_test()
