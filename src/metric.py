from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        # 상태 초기화
        self.add_state('TP', default=torch.tensor(0), dist_reduce_fx='sum') # TP: 맞춤, P; 올바르게 예측된 개수
        self.add_state('FP', default=torch.tensor(0), dist_reduce_fx='sum') # FP: 틀림, P; 잘못 예측된 개수
        self.add_state('FN', default=torch.tensor(0), dist_reduce_fx='sum') # FN: 틀림, N; 예측되지 않은 개수

    def update(self, preds, target):
        """
        preds: (B x C) 클래스 별 예측 값(score)
        target: (B,) GT값, 클래스는 indices로 표현됨
        """
        # MyAccuracy와 같은 과정; The preds (B x C tensor), so take argmax to get index with highest confidence
        predic = torch.argmax(preds, dim=1)

        # 각 클래스에 대해 TP, FP, FN 계산
        for c in torch.unique(target): # 
            true_class = (target == c)
            predicted_class = (predic == c)
            
            self.TP += torch.sum(true_class & predicted_class)  # TP: P 로  맞게 예측
            self.FP += torch.sum(~true_class & predicted_class) # FP: P 로  잘못 예측
            self.FN += torch.sum(true_class & ~predicted_class) # FN: N으로 잘못 예측

    def compute(self):
        precision = self.TP.float() / (self.TP.float() + self.FP.float() + 1e-8) # 1e-8은 0으로 나눔 방지
        recall = self.TP.float() / (self.TP.float() + self.FN.float() + 1e-8)

        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        predic = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        assert predic.shape == target.shape, "Preds(input 1) and target(input 2)의 shape이 다릅니다."
        
        # [TODO] Cound the number of correct prediction
        correct = torch.sum(predic == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
