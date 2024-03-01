# Реализуйте расчет градиента для функции f(w) = prod_i,j(log_e(log_e(w_ij + 7)) в точке w = [[5, 10], [1, 2]]

import torch

w =torch.tensor ([[5,10],[1,2]], requires_grad=True, dtype=torch.float)

function = torch.prod(torch.log(torch.log(w + 7)))
function.backward()
print(w.grad, '<- gradient')

# Реализуйте градиентный спуск для той же функции f(w) = prod_i,j(log_e(log_e(w_ij + 7))
# Пусть начальным приближением будет w^t=0 = [[5, 10], [1, 2]], шаг градиентного спуска alpha=0.001.
# Чему будет равен w^t=500?

w = torch.tensor([[5, 10], [1, 2]], requires_grad=True, dtype=torch.float)
alpha = 0.001

for _ in range(500):
    function = torch.prod(torch.log(torch.log(w + 7)))
    function.backward()
    w.data -=  alpha * w.grad
    w.grad.zero_()

print(w)

# Перепишите пример, используя torch.optim.SGD

w = torch.tensor([[5, 10], [1, 2]], requires_grad=True, dtype=torch.float)
alpha = 0.001
optimizer = torch.optim.SGD([w], lr=0.001)

for _ in range(500):
    function = torch.prod(torch.log(torch.log(w + 7)))
    function.backward()
    optimizer.step()
    optimizer.zero_grad()

print(w) # Код для самопроверки