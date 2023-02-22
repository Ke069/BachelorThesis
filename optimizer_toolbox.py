# SGD_David Klasse
import torch


class SGD_David(torch.optim.Optimizer):
    def __init__(self, parameters, learning_rate):
        super().__init__(parameters, dict(learning_rate=learning_rate))
        # self.learning_rate = learning_rate

    def step(self):
        # paramgroups um das Netz aufzuteilen und zB mit verschiedenen learning_rates zu belegen
        for group in self.param_groups:
            for p in group["params"]:
                # eigentliches Update bei stochastic gradient descent
                stepSizeAndDirection = (-group["learning_rate"]) * (p.grad)
                # print(type(p.grad))
                # anpassen der Gewichte
                # p.data.add_(-group["learning_rate"],stepSizeAndDirection)
                p.data.add_(stepSizeAndDirection)

                # Update_step = Gewicht - learning_rate*p.grad
        # return loss


# Der SignSGD Step soll aus der grad Variable
class signSGD_David_grad(torch.optim.Optimizer):
    def __init__(self, parameters, learning_rate):
        super().__init__(parameters, dict(learning_rate=learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                # Zwischenschritt hier den Eingabeverktor in p.grad auf sein Vorzeichen reduzieren - wie ?
                # Aaron sagt durch den Absolutwert teilen
                signGrad = p.grad / torch.abs(p.grad)
                print(signGrad)
                stepSizeAndDirection = (-group["learning_rate"]) * (signGrad)
                p.data.add_(stepSizeAndDirection)


# Haendisch erstmal aus grad_batch erstellen
# class signSGD_David_grad_batch(torch.optim.Optimizer):
#    def __init__(self, parameters, learning_rate):
#        super().__init__(
#            parameters,
#            dict(learning_rate = learning_rate))

#    def step(self):
#        for group in self.param_groups:
#            for p in group["params"]:
#                #Hier soll der signGrad via grad_batch berechnet werden
#                # Muss dazu
#                stepSizeAndDirection = (-group["learning_rate"])*
