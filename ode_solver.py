import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt

def imp_rk_step(func, x, t, h):
    k1 = func(x, t)
    k2 = func(tuple(x_+h/3*k1_ for x_, k1_ in zip(x, k1)), t+h/3)
    k3 = func(tuple(x_+2*h/3*(k1_/-3+k2_)
                    for x_, k1_, k2_ in zip(x, k1, k2)), t+2*h/3)
    k4 = func(tuple(x_+h*(k1_-k2_+k3_)
                    for x_, k1_, k2_, k3_ in zip(x, k1, k2, k3)), t+h)
    return tuple(h/8*(k1_+3*k2_+3*k3_+k4_) for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


class Solver(object):
    def __init__(self, func, h=None):
        self.func = func
        self.h = h
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float64

    def step_func(self):
        pass

    def integrate(self, x0, T):
        # take tensors return tensors
        # tale tuple then return tuple
        t0 = T[0].item()
        t1 = T[-1].item()

        h = self.h
        if h is not None:
            N = math.ceil((abs(t1 - t0)/h))+1
            t_grid = torch.arange(0, N).to(
                device=self.device, dtype=self.dtype) * h + t0
            if t_grid[-1] > t1:
                t_grid[-1] = t1

        else:
            t_grid = T

        hist = [x0]
        for t0, t1 in zip(t_grid[:-1], t_grid[1:]):
            h = t1-t0
            dx = self.step_func(self.func, x0, t0, h)
            x1 = [x0_+dx_ for x0_, dx_ in zip(x0, dx)]
            
            hist.append(tuple(x1))
            x0 = x1
        return x1, tuple(map(torch.stack, tuple(zip(*hist))))


def odeint(func, x0, T, h=None):
    # tensor input then tensor output 

    t0 = T[0]
    t1 = T[-1]
    if t1 < t0:
        _t = t1
        t1 = t0
        t0 = _t
        _base_reverse_func = func
        func= lambda x, t: -_base_reverse_func(x, t)
    tensor_input = False


    if torch.is_tensor(x0):
        tensor_input = True
        x0 = (x0,)
        _base_func = func
        func= lambda x,t: tuple(_base_func(x[0], t))

    solver = RK4(func, h=h)    
    x1, hist = solver.integrate(x0, T)
    if tensor_input:
        x1= x1[0]
        hist = hist[0]
    return x1, hist



class RK4(Solver):
    def step_func(self, func, x0, t0, h):
        return imp_rk_step(func, x0, t0, h)




class adjoint_method(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, h, x0, t, flat_params):
        # input x0 as tuple
        ctx.func= func
        ctx. h=h
        
        with torch.no_grad():
            x, solution = odeint(func, x0, t, h)
            # solution is a tuple with a tensor of shape timestep, *xo_shape
        ctx.save_for_backward(solution, t, flat_params)
        return x

    @staticmethod
    def backward(ctx,dLdx):
        ans, t, flat_params = ctx.saved_tensors
        # ans has the size (T, batchsize, x_shape)
        # solution is list with tensors now 
        func = ctx.func
        h = ctx.h
        f_params=func.parameters()
        # T = t.size()[0]
        T = ans.shape[0] 

        def aug_dynamics(aug_x_i, t_i):
            # aug_x_i has the form of tuple(x_i, *a)
            # x_i should be a tensor and a_aug hat the same dimension as x_i  we ignore a_t and a_p here?
            x_i= aug_x_i[0]
            a_aug = aug_x_i[1]
            
            with torch.set_grad_enabled(True):
                
                x_i= x_i.detach().requires_grad_(True)
                t_i=t_i.to(x_i.device).detach().requires_grad_(True)
                
                f_eval=func(x_i,t_i)

                # f_eval is tensor                
                # a_aug = [a, a_t, a_p]
                # hier is we compute -a_aug^T * d f(x,t)/d [x,t,params]
                adfdx, adfdt, *adfdp = torch.autograd.grad(
                    (f_eval,), (x_i,t_i)+tuple(f_params) , tuple(a_aug), allow_unused=True, retain_graph=True
                )


            
            #  set the gradients to zero if there is no gradient (.grad() returns None)
            


            adfdx = torch.zeros_like(x_i) if adfdx is None else adfdx
            # hier x_ has the shape batchsize, *x_shape
            if len(f_params) == 0:
                adfdp = torch.tensor(0.).to(x_i)
            else:
                adfdp= torch.cat([adfdp_.flatten() if adfdp_ is not None else torch.zero_like(p_) for adfdp_, p_ in zip(adfdp, flat_params)])
            
            return (f_eval, -adfdx, -adfdt, -adfdp) 
            
        with torch.no_grad():
            adj_x = dLdx[-1]
            adj_p = torch.zeros_like(flat_params).to(dLdx)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(T, 1).to(t)

            for i in range(T-1, 0 - 1):
                # dLdx is a list with dLdx(t_i) at i th place
                dLdx_i = dLdx[i]
                t_i=t[i]
                x_i = ans[i]
                f_i = func(x_i,t_i)
                dLdt_i = sum(
                    torch.dot(f_i.reshape(-1),dLdx_i.reshape(-1)).reshape(1)
                )
                
                adj_t[i] = adj_t[i] - dLdt_i
                aug_x = (x_i, adj_x, adj_t[i], adj_p)
                aug_solution, _ = odeint(
                    aug_dynamics, aug_x, torch.tensor([t[i], t[i-1]],h)
                ) 
                adj_x=aug_solution[-3]
                adj_t[i-1] = aug_solution[-2]
                adj_p = aug_solution[-1]
                
                adj_x= adj_x+dLdx[i-1]
                del aug_x, aug_solution
            return None, None, adj_x, adj_t, adj_p
    

def odeint_adjoint(func, x0, t, h = None):
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    if torch.is_tensor(x0):
        class TupleFunc(nn.Module):
            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func
            def forward(self, x, t):
                return self.base_func(x, t)

    func = TupleFunc(func)
    f_params= func.parameters()
    flat_params = torch.cat([p_.flatten() for p_ in f_params])
    x = adjoint_method.apply(func, h, x0, t, flat_params)
    return x
    










        


    










if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

    x0 = torch.tensor([[2., 0.]]).to(device)
    t0=0.
    t1=1.
    T=torch.tensor([t0,t1])
    t = torch.arange(t0, t1+0.03, 0.03)

    class ODE(nn.Module):
        def forward(self, x, t):
            return torch.mm(x**3, A)

    with torch.no_grad():
        solution, hist= odeint(ODE(), x0, T, 0.03)

 
    print(solution.shape)
    print(hist.shape)

    t=t.cpu()
    hist = hist.cpu()
    fig = plt.figure(figsize=(4, 4), facecolor='white')
    ax_traj = fig.add_subplot(111, frameon=False)
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.numpy(), hist.numpy()[
        :, 0, 0], t.numpy(), hist.numpy()[:, 0, 1], 'g-')

    # ax_traj.set_ylim(-2, 2)
    ax_traj.legend()

    plt.show()
