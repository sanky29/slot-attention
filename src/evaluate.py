class Evaluator:

    def __init__(self, args, model):
        self.args = args
        self.model = model
        
    def evaluate(self, data):
        y_final = torch.Tensor
        for data_hypothesis, data_premise in tqdm(data):
            y_pred = self.model(data_premise, data_hypothesis)
            y_pred = torch.argmax(y_pred, axis = 1)
            y_final = torch.cat(y_final, y_pred)
        return y_final