import torch
import torch.nn as nn
import random
import math

def hinge_loss(scores, targets):
    '''
    multi-class extension: https://en.wikipedia.org/wiki/Hinge_loss
    Weston Jason & Watkins Chris (1999). 
    Support Vector Machines for Multi-Class Pattern Recognition. 
    European Symposium on Artificial Neural Networks.
    '''
    batch_size, num_classes = scores.shape
    assert batch_size == targets.shape[0]
    correct_class_score = scores[range(batch_size), targets].view(-1, 1)
    scores = scores - correct_class_score + 1
    scores[scores<=0] = 0
    scores[range(batch_size), targets] = 0
    return scores.sum()
    
class LinearSVM():
    def __init__(self, tol=1e-4, C=1.0, verbose=False, max_iter=1000,       # accord to scikit learn
            lr=1e-2):                                                       # specific params for pytorch
        self.W = None
        self.tol = tol
        self.C = C
        self.verbose = verbose
        self.max_iter = max_iter
        
        self.lr = lr
        
    def fit(self, X, y, batch_size=1024):
        # Config
        num_data, num_feat = X.shape
        num_classes = int(y.max().item()) + 1
        num_batches = math.ceil(num_data/batch_size)

        if self.W is None:
            self.W = nn.Linear(num_feat, num_classes)
        
        optimizer = torch.optim.SGD(
            self.W.parameters(), lr=self.lr, weight_decay=self.C)
            
        # Train
        prev_loss = 0.
        idx = list(range(num_data))
        for epoch in range(self.max_iter):
            random.shuffle(idx)
            tot_loss = 0.
            self.W.train()
            for batch_idx in range(num_batches):
                optimizer.zero_grad()
                cur_idx = idx[batch_size*(batch_idx) : batch_size*(batch_idx+1)]
                batch = X[cur_idx]
                targets = y[cur_idx]
                pred = self.W(batch) # batch_size num_classes
                loss = hinge_loss(pred, targets)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            
            tot_loss /= num_data
            acc = self.score(X, y)
            if self.verbose:
                print('>>> iter %d, loss %.4f, acc %.4f'%(epoch+1, tot_loss, acc))
            if math.fabs(tot_loss-prev_loss) < self.tol:
                print('early stop, prev loss: %.4f, curr loss: %.4f'%(prev_loss, tot_loss))
                break
            else:
                prev_loss = tot_loss
    
    def predict(self, X):
        self.W.eval()
        scores = self.W(X)
        return torch.argmax(scores, dim=1)

    def score(self, X, y):
        self.W.eval()
        batch_size = X.shape[0]
        scores = self.W(X)
        preds = torch.argmax(scores, dim=1)
        acc = (preds==y).sum().item()/batch_size
        return acc


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=2000, n_features=4, random_state=0)
    X_train, X_test = torch.from_numpy(X[:1500]).float(), torch.from_numpy(X[1500:]).float()
    y_train, y_test = torch.from_numpy(y[:1500]).long(), torch.from_numpy(y[1500:]).long()
    print(y[:20])
    print(X.shape, y.shape)
    model = LinearSVM(verbose=True)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    