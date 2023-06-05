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
                
               
                #### VARIANCE ADAPTATION ####
                m_A = (1/len(p.grad_batch))* torch.sum(p.grad_batch, axis=0)

                v_A = (1/len(p.grad_batch))* torch.sum(p.grad_batch**2, axis=0)
            
                Gradient_with_VarianceAdaptation = torch.sqrt(1/(1+((v_A-m_A**2)/m_A**2))) * torch.sign(p.grad)

               
                stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation)
                p.data.add_(stepSizeAndDirection)
                

class signSGD_David_grad_batch(torch.optim.Optimizer):
    def __init__(self, parameters, learning_rate):
        print("Jetzt wird signSGD_David_grad_batch verwendet")
        super().__init__(
            parameters,
            dict(learning_rate = learning_rate))

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
           
                signGrad = torch.sign(torch.sum(p.grad_batch, axis=0))
                ### Check ####
                our_grad = torch.sum(p.grad_batch, axis=0)
                pytorch_grad = p.grad
        
                ####Variance Adaptation #####
                m_A = 1/len(p.grad_batch)*torch.sum(p.grad_batch, axis=0)
               
                v_A = 1/len(p.grad_batch)*torch.sum((p.grad_batch)**2, axis=0)
               
                #Wurzel aus Balles Paper
                Gradient_with_VarianceAdaptation = torch.sqrt(1/1+((v_A-(m_A)**2)/(m_A)**2))

                stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation)*(signGrad) 
                p.data.add_(stepSizeAndDirection) 

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
                stepSizeAndDirection = (-group["learning_rate"])*(Gradient_with_VarianceAdaptation) * (vectorized_ergebnisliste)

                # wihtout variance adaptation
                #stepSizeAndDirection = -group["learning_rate"] * vectorized_ergebnisliste
                
                # APPLY GRADS
                p.data.add_(stepSizeAndDirection)        

           



