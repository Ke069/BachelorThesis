# SGD_David Klasse 
import torch
import torch.nn as nn

class SGD_David(torch.optim.Optimizer):
    def __init__(self, parameters, learning_rate):
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))
        #self.learning_rate = learning_rate
    def step(self):
        #paramgroups um das Netz aufzuteilen und zB mit verschiedenen learning_rates zu belegen
        for group in self.param_groups:
            for p in group["params"]:
                #eigentliches Update bei stochastic gradient descent
                stepSizeAndDirection = (-group["learning_rate"])*(p.grad)
                #print(type(p.grad))
                #anpassen der Gewichte
                #p.data.add_(-group["learning_rate"],stepSizeAndDirection)
                p.data.add_(stepSizeAndDirection)

                #Update_step = Gewicht - learning_rate*p.grad
        #return loss


# Der SignSGD Step soll aus der grad Variable 
# Hier der einfache 
class signSGD_David_grad(torch.optim.Optimizer):
    print("Jetzt wird signSGD_David_grad verwendet ")
    def __init__(self,parameters, learning_rate):
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                #Zwischenschritt hier den Eingabeverktor in p.grad auf sein Vorzeichen reduzieren - wie ?
               
                signGrad = p.grad/torch.abs(p.grad)
                print(signGrad)
                stepSizeAndDirection = (-group["learning_rate"])*(signGrad)
                p.data.add_(stepSizeAndDirection)
                #print("Jetzt wird signSGD_David_grad verwendet ")



#Haendisch erstmal aus grad_batch erstellen
class signSGD_David_grad_batch(torch.optim.Optimizer):
    print("Jetzt wird signSGD_David_grad_batch verwendet")
    def __init__(self, parameters, learning_rate):
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                #Hier soll der signGrad via grad_batch berechnet werden
                # grad_batch Variable aufsummieren , von der Summe das Vorzeichen holen.
                # bei einem Wert mit Absolutwert  
                signGrad = (sum(p.grad_batch)/torch.abs(sum(p.grad_batch)))
                #Lukas
                #signGrad = (sum(p.grad_batch)/torch.sign(p.grad_batch))
                #Da sum p.grad_batch und p.grad dasselbe sein sollten
                #signGrad = (sum(p.grad_batch)/torch.abs(p.grad))
                stepSizeAndDirection = (-group["learning_rate"])*(signGrad) 
                #print("Jetzt wird signSGD_David_grad_batch verwendet")   




#class majorityVoteSignSGD(torch.optim.Optimizer):
#    def __init__(self,parameters, learning_rate):
#        super().__init__(
#            parameters,
#            dict(learning_rate = learning_rate))
#
#    def step(self):
#        for group in self.param_groups:
#            for p in group["params"]:
#                #Zwischenschritt hier den Eingabeverktor in p.grad auf sein Vorzeichen reduzieren - wie ?
                
                #Quest1 wie kommme ich an die Vorzeichen von param.grad_batch 
                #-> np.sign(grad_batch)
                #Quest2 zählen der plusse und minusse
                #Quest3 - die Entscheidung
                # 
#                signGrad = p.grad/torch.abs(p.grad)
#                print(signGrad)
#                stepSizeAndDirection =(-group["learning_rate"])*(signGrad)
#                p.data.add_(stepSizeAndDirection)


