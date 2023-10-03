class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.lowest_loss = None
        self.epochs_stagnant = 0

    def consume(self, new_loss):
        if self.lowest_loss is None:
            self.lowest_loss = new_loss
            self.epochs_stagnant = 0
        elif self.lowest_loss <= new_loss:
            self.epochs_stagnant += 1
        else:
            self.lowest_loss = new_loss
            self.epochs_stagnant = 0

    def should_stop(self):
        return self.epochs_stagnant >= self.patience
