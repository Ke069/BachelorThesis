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
    
    def __init__(self,parameters, learning_rate):
        print("Jetzt wird signSGD_David_grad verwendet ")
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                #Zwischenschritt hier den Eingabeverktor in p.grad auf sein Vorzeichen reduzieren - wie ?
               
                #print("Gradient = ", p.grad)

                #Adding Variance Adaptation
                m_A = 1/len(p.grad_batch)-torch.sum(p.grad_batch, axis=0)
                #print(m_A)
                v_A = 1/len(p.grad_batch)-torch.sum((p.grad_batch)**2, axis=0)
                #print(v_A)

                Gradient_with_VarianceAdaptation = torch.sqrt(1/1+((v_A-(m_A)**2)/(m_A)**2)) * torch.sign(p.grad)

                signGrad = torch.sign(p.grad)  # p.grad/torch.abs(p.grad)
                #print("Sign(Gradient) = ", signGrad)
                stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation)
                p.data.add_(stepSizeAndDirection)
                #print("Jetzt wird signSGD_David_grad verwendet ")




#Haendisch erstmal aus grad_batch erstellen
class signSGD_David_grad_batch(torch.optim.Optimizer):
    def __init__(self, parameters, learning_rate):
        print("Jetzt wird signSGD_David_grad_batch verwendet")
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                #Hier soll der signGrad via grad_batch berechnet werden
                # grad_batch Variable aufsummieren , von der Summe das Vorzeichen holen.
                # bei einem Wert mit Absolutwert
                #print("Grad_Batch Variable ", p.grad_batch)
                #print("\n", p.grad_batch.shape)

                # signGrad = (sum(p.grad_batch)/torch.abs(sum(p.grad_batch)))
                # print(sum(p.grad_batch).shape)
                signGrad = torch.sign(torch.sum(p.grad_batch, axis=0))
                #print(signGrad.shape)


                our_grad = torch.sum(p.grad_batch, axis=0)
                pytorch_grad = p.grad
                #print("\nour_grad = ", our_grad)
                #print("pytorch_grad = ", pytorch_grad)
                #print("Tensors close?", torch.allclose(our_grad, pytorch_grad))

                ####Variance Adaptation #####
                m_A = 1/len(p.grad_batch)-torch.sum(p.grad_batch, axis=0)
                #print(m_A)
                v_A = 1/len(p.grad_batch)-torch.sum((p.grad_batch)**2, axis=0)
                #print(v_A)
                #Wurzel aus Balles Paper
                Gradient_with_VarianceAdaptation = torch.sqrt(1/1+((v_A-(m_A)**2)/(m_A)**2))

                
                
                #signGrad = (sum(p.grad_batch)/torch.sign(p.grad_batch))
                #Da sum p.grad_batch und p.grad dasselbe sein sollten
                #signGrad = (sum(p.grad_batch)/torch.abs(p.grad))
                stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation)*(signGrad) 
                #print("Jetzt wird signSGD_David_grad_batch verwendet")   

class majorityVoteSignSGD(torch.optim.Optimizer):
    def __init__(self,parameters, learning_rate):
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                #Zwischenschritt hier den Eingabeverktor in p.grad auf sein Vorzeichen reduzieren - wie ?
                #signGrad = torch.sign((torch.sum(p.grad_batch, axis=0))

                grad_batch_sign = torch.sign(p.grad_batch)
                #print("Sign von GradBatch", grad_batch_sign)
                #print("Grad_Batch von Backpack ", p.grad_batch)
                #print("Grad Variable von SGD ",p.grad)


                ##Container 
                # scheint grad_batch_sign
                is_one = torch.sum(grad_batch_sign == 1, axis=0)
                #print("is_one: ", is_one)
                is_zero = torch.sum(grad_batch_sign == 0, axis=0)
                #print("is_zero: ", is_zero)
                is_minus_one = torch.sum(grad_batch_sign == -1, axis=0)
                #print("is_minus_one: ", is_minus_one)

                r = is_one
                s = is_zero
                t = is_minus_one

                stacked = torch.stack([r,s,t], dim = 0)
                #print("Stacked ", stacked)
                ##### Variance Adaptation ###
                m_A = 1/len(p.grad_batch)-torch.sum(p.grad_batch, axis=0)
                #print(m_A)
                v_A = 1/len(p.grad_batch)-torch.sum((p.grad_batch)**2, axis=0)
                #print(v_A)
                #Wurzel aus Balles Paper
                Gradient_with_VarianceAdaptation = torch.sqrt(1/1+((v_A-(m_A)**2)/(m_A)**2)) 

 

                a = torch.argmax(stacked, 0).long()
                vectorized_ergebnisliste = torch.Tensor([1,0,-1])[a]
                #print("mjv Gradient: ", vectorized_ergebnisliste)
                #stepSizeAndDirection = (-group["learning_rate"])*(vectorized_ergebnisliste)

                stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation) * (vectorized_ergebnisliste)
                
                #Quest1 wie kommme ich an die Vorzeichen von param.grad_batch 
                #-> np.sign(grad_batch)
                #Quest2 zählen der plusse und minusse
                #Quest3 - die Entscheidung
                # 
#                signGrad = p.grad/torch.abs(p.grad)
#                print(signGrad)
#                stepSizeAndDirection =(-group["learning_rate"])*(signGrad)
#                p.data.add_(stepSizeAndDirection)

#Logik von example_clean hierin uebersetzen


