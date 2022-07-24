import numpy as np
import matplotlib.pyplot as plt
import torch


def k(x, y, gamma=2):
    n, d1 = x.shape
    m, d2 = y.shape
    assert d1 == d2
    x2 = torch.einsum("ij,ij->i", x, x).view(-1, 1)
    y2 = torch.einsum("ij,ij->i", y, y).view(1, -1)
    xy = torch.einsum("ij,kj->ik", x, y)
    x2 = x2.repeat(1, m)
    y2 = y2.repeat(n, 1)
    norm2 = x2 - 2 * xy + y2
    return torch.exp(-gamma * norm2).sum() / (n * m)


def loss_fun(x0_t, xk_t, parameters, gamma=1):
    all_parameters = [x0_t] + parameters + [xk_t]
    loss = 0
    for i in range(len(all_parameters) - 1):
        loss += (
            k(all_parameters[i + 1], all_parameters[i + 1], gamma)
            - 2 * k(all_parameters[i + 1], all_parameters[i], gamma)
            + k(all_parameters[i], all_parameters[i], gamma)
        )
    return loss

n = 1000
k_num = 8
gamma = 1/32

x0_cand = np.stack(
    [
        np.random.multivariate_normal((-5, 0), np.eye(2), n),
        np.random.multivariate_normal((5, 0), np.eye(2), n),
    ]
)
xk_cand = np.stack(
    [
        np.random.multivariate_normal((0, -5), np.eye(2), n),
        np.random.multivariate_normal((0, 5), np.eye(2), n),
    ]
)

idx = np.random.choice(np.arange(2), size=(n,))
xk = np.concatenate([xk_cand[0, idx == 0, :], xk_cand[1, idx == 1, :]])
x0 = np.concatenate([x0_cand[0, idx == 0, :], x0_cand[1, idx == 1, :]])

x0_t = torch.tensor(x0)
xk_t = torch.tensor(xk)

parameters = [
    torch.tensor(
        np.random.multivariate_normal((0, 0), np.eye(2), n), requires_grad=True
    )
    for _ in range(k_num)
]

optim = torch.optim.Adam(parameters, lr=0.1)
loss_history = []
history = []
all_parameters = [x0_t] + parameters + [xk_t]


for t in range(201):
    if t % 100 == 0:
        print(t)
        h = [par.detach().numpy().copy() for par in parameters]
        history.append(h)

    loss = loss_fun(x0_t, xk_t, parameters, gamma=gamma)
    loss.backward()
    optim.step()
    optim.zero_grad()

    if t % 100 == 0:
        loss_history.append(loss.item())

fig, axes = plt.subplots(1, 10, figsize=(20, 2))
h_ext = [x0_t.numpy()] + history[-1] + [xk_t.numpy()]
for j, x in enumerate(h_ext):
    axes[j].scatter(x[:, 0], x[:, 1], alpha=0.1, linewidths=0)
    axes[j].set_xlim([-8, 8])
    axes[j].set_ylim([-8, 8])
    axes[j].axis("off")
plt.savefig('interpolation.pdf')
plt.show()

