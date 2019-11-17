import torch
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt

# class Solver(object):
#     def __init__( self, func, h=None):
#         self.func= func
#         self.h = h
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.dtype = torch.float64

#     def step_func(self):
#         pass

#     def integrate(self, x0, T):
#         #take tensors as input 
#         t0=T[0].item()
#         t1=T[-1].item()



#         h=self.h
#         if h is not None:
#             N = math.ceil((abs(t1 - t0)/h))+1
#             t_grid = torch.arange(0, N).to(device=self.device, dtype=self.dtype) * h + t0
#             if t_grid[-1] > t1:
#                 t_grid[-1] = t1

#         else:
#             t_grid =T.to(device=self.device, dtype=self.dtype)

#         hist = [x0]
#         print(t_grid)
#         for t0, t1 in zip(t_grid[:-1], t_grid[1:]):
#             h=t1-t0
#             dx= self.step_func(self.func, x0, t0, h)
#             x1= x0+dx
#             hist.append(x1)
#             x0=x1
#             # print(type(x0))
#             # print(hist[-1].shape)
#             #alternativ: x = torch.zeros(len(t_grid), batchsize, *x_shape).to(x0)
#         # return a tuple (tuple(), tensor)
#         return x1, torch.stack(hist)


# class RK4(Solver):
#     def step_func(self, func, x0, t0, h):
#         return imp_rk_step(func, x0, t0, h)

# def imp_rk_step(func, x, t, h):
#     k1= func(x, t)
#     k2= func(x+h/3*k1, t+h/3)
#     k3= func(x+2/3*h*(k1/-3 +k2), t+2/3*h)
#     k4= func(x+h*(k1-k2+k3) ,t+h)
#     return h/8*(k1 + 3*k2 + 3*k3 + k4)


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
            t_grid = T.to(device=self.device, dtype=self.dtype)

        hist = [x0]
        print(t_grid)
        for t0, t1 in zip(t_grid[:-1], t_grid[1:]):
            h = t1-t0
            dx = self.step_func(self.func, x0, t0, h)
            x1 = [x0_+dx_ for x0_, dx_ in zip(x0, dx)]
            
            hist.append(tuple(x1))
            x0 = x1
            # print(type(x0))
            # print(hist[-1].shape)
            #alternativ: x = torch.zeros(len(t_grid), batchsize, *x_shape).to(x0)
        # return a tuple (tuple(), tensor)
        # if len(x1)==1:
        #     return x1, torch.stack(hist)
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
        def func(x, t): return -_base_reverse_func(x, t)
    # if isinstance(x0, tuple):
    #     tuple_input = True
    #     print('tuple_input')
    #     func= lambda x,t: tuple(_base_func(x[0], t))

    if torch.is_tensor(x0):
        tensor_input = True
        print('tensor_input')
        
        # turn x0 into a tuple which has x0 as its element
        x0 = (x0,)
        _base_func = func
        # print('TupleFunc x is a {} and has the len {}'.format(type(x),len(x)))
        func= lambda x,t: tuple(_base_func(x[0], t))

    # assert isinstance(x0, tuple), 'x0 has to be a torch.tensor or a tuple'
    # for x0_ in x0:
    #     assert torch.is_tensor(x0_), 'each element must be a torch.Tensor but received {}'.format(type(x0_))
    #     if not torch.is_floating_point(x0_):
    #         raise TypeError(f'y0 is not a floating point tensor but a {type(x0)}')

    solver = RK4(func, h=h)
    
    # if tensor_input:
    #     hist = hist[0]
    #     return x1[0], hist
    if tensor_input:
        x1, hist = solver.integrate(x0, T)
        return x1[0], hist[0]
    return x1, hist



class RK4(Solver):
    def step_func(self, func, x0, t0, h):
        return imp_rk_step(func, x0, t0, h)


# def odeint(func, x0, T, h=None):
#     # only take tensor as input

#     t0=T[0]
#     t1=T[-1]
#     if t1<t0:
#         _t = t1
#         t1 = t0
#         t0 = _t
#         _base_reverse_func = func
#         func = lambda x, t:-_base_reverse_func(x, t)
#     # if isinstance(x0, tuple):
#     #     tuple_input = True
#     #     print('tuple_input')
#     #     func= lambda x,t: tuple(_base_func(x[0], t))

#     # if torch.is_tensor(x0):
#     #     tensor_input = True
#     #     print('tensor_input')

#         # # turn x0 into a tuple which has x0 as its element
#         # x0 = (x0,)
#         # _base_func = func
#         # # print('TupleFunc x is a {} and has the len {}'.format(type(x),len(x)))
#         # func= lambda x,t: tuple(_base_func(x[0], t))
    
#     # assert isinstance(x0, tuple), 'x0 has to be a torch.tensor or a tuple'
#     # for x0_ in x0:
#     #     assert torch.is_tensor(x0_), 'each element must be a torch.Tensor but received {}'.format(type(x0_))
#     #     if not torch.is_floating_point(x0_):
#     #         raise TypeError(f'y0 is not a floating point tensor but a {type(x0)}')
    
#     assert torch.is_tensor(x0), 'x0 has to be a torch.tensor'
    
#     solver = RK4(func, h = h)
#     x1, hist = solver.integrate(x0,T)
    
#     # if tensor_input: 
#     #     hist = hist[0]
#     #     return x1[0], hist 
#     return x1, hist
        


class adjoint_method(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, h, x0, t, flat_params):
        # input x0 as tuple

        # if torch.is_tensor(x0):
        #     batchsize, *x_shape = x0.shape
        # elif isinstance(x0,tuple):
        #     x0_len=len(x0)
        #     batchsize, *x_shape = x0[0].shape
        
        timestep= t.size(0)
        
        # print('forward input x0 has the type {} and len {}'.format(type(x0),len(x0)))
        
        ctx.func= func
        ctx. h=h
        
        with torch.no_grad():
            x, solution = odeint(func, x0, t, h)
            # solution is a tuple with a tensor of shape timestep, *xo_shape
        print('forward() solution if of the the type {} and has the shape {}'.format(type(solution), solution.shape))
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
        T, batchsize, *x_shape = ans.shape

        def aug_dynamics(aug_x_i, t_i):
            # aug_x_i = tuple(x_i, *a)
            
            # x_i should be a tensor and a_aug hat the same dimension as x_i  we ignore a_t and a_p here?
            x_i= aug_x_i[0]
            a_aug = aug_x_i[1]
            # flat_params = [p_.flatten() for p_ in f_params]


            
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


                # adfdx = adfdx.to(x_i[0]) if adfdx is not None else torch.zeros_like(y_) if
            
            #  set the gradients to zero if there is no gradient (.grad() returns None)
            


            adfdx = torch.zeros_like(x_i) if adfdx is None else adfdx
            # hier x_ has the shape batchsize, *x_shape
            
            # first flatten f_param? 
            if len(f_params) == 0:
                adfdp = torch.tensor(0.).to(x_i)
            else:
                adfdp= torch.cat([adfdp_.flatten() if adfdp_ is not None else torch.zero_like(flat_p_) for adfdp_, p_ in zip(adfdp, flat_params)])
            
            # # now flatten f_eval, adfdx, adfdt...
            # f_eval = f_eval.view(batchsize, n_dim)
            # adfdx = adfdx.view(batchsize, n_dim)

            # f_eval -adfdx -adfdt are tensors -adfdp a flat tensor
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
                        # torch.bmm(torch.transpose(f_i,1,2),dLdx_i)
                        torch.dot(f_i.reshape(-1),dLdx_i.reshape(-1)).reshape(1)
                    )
                    print('dLdt_i is of the the type {} and has the shape {}'.format(
                        type(dLdt_i), dLdt_i.shape))
                    
                    adfdt[i] = adj_t[i] - dLdt_i
                    aug_x = (x_i, adj_x, adj_t[i], adj_p)
                    aug_solution, _ = odeint(
                        aug_dynamics, aug_x, torch.tensor([t[i], t[i-1]],h)
                    ) 


                    adj_x=aug_solution[-3]
                    adfdt[i-1]= aug_solution[-2]
                    adfdp = aug_solution[-1]
                    
                    adfdx= adfdx+dLdx[i-1]
                    
                    del aug_x, aug_solution
            return None, None, adfdx, adfdt, adfdp
        
            # # flatten adfdp (.view(-1))
            # # adfdp has now a tensor of the form tensor.size([number of params])

            # adfdt = adfdt.to(x_i) if adfdt is not None else torch.zeros_like(t_i).to(x_i)
            
            # # f_eval a tuple adfdx a tuple, adfdt and adfdp are tensors
            # return (*f_eval, *adfdx, adfdt, adfdp)
    

# def odeint_adjoint(func, x0, T, h = None):
    










        


    










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

    # x0=(x0,)
    # func = lambda x,t : tuple(ODE()(x[0],t))
    # # print(func(x,t))
    # solver=RK4(func, x0, h = None)
    # solution = solver.integrate(interval)
    # solution = solution[0]
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
