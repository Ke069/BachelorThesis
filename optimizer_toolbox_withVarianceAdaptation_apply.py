# SGD_David Klasse 
import torch
#import torch.nn as nn

#from backpack import backpack, extend
#from backpack.extensions import BatchGrad


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
                
                stepSizeAndDirection = (-group["learning_rate"])*(p.grad)
                p.data.add_(stepSizeAndDirection)






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
                m_A = (1/len(p.grad_batch))* torch.sum(p.grad_batch, axis=0)
                #print(m_A)
                v_A = (1/len(p.grad_batch))* torch.sum(p.grad_batch**2, axis=0)
                #print(v_A)
                 

                Gradient_with_VarianceAdaptation = torch.sqrt(1/(1+((v_A-m_A**2)/m_A**2))) * torch.sign(p.grad)

                #signGrad = torch.sign(p.grad)  # p.grad/torch.abs(p.grad)
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
                m_A = 1/len(p.grad_batch)*torch.sum(p.grad_batch, axis=0)
                #print(m_A)
                v_A = 1/len(p.grad_batch)*torch.sum((p.grad_batch)**2, axis=0)
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
                #signGrad = torch.sign((torch.sum(p.grad_batch, axis=0))

                #magnitude = torch.abs(p.grad_batch).mean(dim=0)
                grad_batch_sign = torch.sign(p.grad_batch)
              


                is_one = torch.sum(grad_batch_sign == 1, axis=0)
                #is_one = torch.sum(grad_batch_sign > 0.5, axis=0)
                
                is_minus_one = torch.sum(grad_batch_sign == -1, axis=0)
                #is_minus_one = torch.sum(grad_batch_sign < -0.5, axis=0)
                
                is_zero = torch.sum(grad_batch_sign == 0, axis=0)
                #is_zero = 1 - is_one + is_minus_one
                

                r = is_one
                s = is_zero
                t = is_minus_one

                stacked = torch.stack([r,s,t], dim = 0)
           
                m_A = (1/len(p.grad_batch))* torch.sum(p.grad_batch, axis=0)
                
                v_A = (1/len(p.grad_batch))* torch.sum(p.grad_batch**2, axis=0)
              
                
                eps = 1e-8

                denominator = 1 + (v_A - m_A**2)/(m_A**2 + eps)
                Gradient_with_VarianceAdaptation = torch.sqrt(
                    1/denominator # must be greater 0 -> otherwise None?
                ) + eps

                a = torch.argmax(stacked, 0).long()
                vectorized_ergebnisliste = torch.Tensor([1,0,-1])[a]
                
                

                # with variance adaptation
                #stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation) * (vectorized_ergebnisliste)

                # wihtout variance adaptation
                stepSizeAndDirection = -group["learning_rate"] * vectorized_ergebnisliste
                
                # APPLY GRADS
                p.data.add_(stepSizeAndDirection)        

           



