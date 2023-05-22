import torch
torch.set_grad_enabled(False)


def orthogonal_projection(s, w, device='cuda'):
    """Orthogonally project the (n+1)-dimensional vectors w onto the tangent space T_sS^n.

    Args:
        s (torch.Tensor): point on S^n
        w (torch.Tensor): batch of (n+1)-dimensional vectors to be projected on T_sS^n

    Returns:
        Pi_s(w) (torch.Tensor): orthogonal projections of w onto T_sS^n

    """
    # Get dimensionality of the ambient space (dim=n+1)
    dim = s.shape[0]

    # Calculate orthogonal projection
    I_ = torch.eye(dim, device=device)
    P = I_ - s.unsqueeze(1) @ s.unsqueeze(1).T

    return w.view(-1, dim) @ P.T


def logarithmic_map(s, q, epsilon=torch.finfo(torch.float32).eps):
    """Calculate the logarithmic map of a batch of sphere points q onto the tangent space TsS^n.

    Args:
        s (torch.Tensor): point on S^n defining the tangent space TsS^n
        q (torch.Tensor): batch of points on S^n
        epsilon (uint8) : small value to prevent division by 0

    Returns:
        log_s(q) (torch.Tensor): logarithmic map of q onto the tangent space TsS^n.

    """
    torch._assert(len(s.size()) == 1, 'Only a single point s on S^n is supported')
    dim = s.shape[0]
    q = q.view(-1, dim)  # ensure batch dimension
    q = q / torch.norm(q, p=2, dim=-1, keepdim=True)  # ensure unit length

    pi_s_q_minus_s = orthogonal_projection(s, (q - s))

    return (torch.arccos(torch.clip((q * s).sum(axis=-1), -1.0, 1.0)).unsqueeze(1)) * pi_s_q_minus_s / \
        (torch.norm(pi_s_q_minus_s, p=2, dim=1, keepdim=True) + epsilon)


def exponential_map(s, q):
    """Calculate the exponential map at point s for a batch of points q in the tangent space TsS^n.

    Args:
        s (torch.Tensor): point on S^n defining the tangent space TsS^n
        q (torch.Tensor): batch of points in TsS^n.

    Returns:
        exp_s(q) (torch.Tensor): exponential map of q from points in the tangent space TsS^n to S^n.

    """
    torch._assert(len(s.size()) == 1, 'Only a single point s on S^n is supported')
    dim = s.shape[0]
    q = q.view(-1, dim)  # ensure batch dimension

    q_norm = torch.norm(q, p=2, dim=1).unsqueeze(1)
    out = torch.cos(q_norm) * s + torch.sin(q_norm) * q / q_norm
    return out / torch.norm(out, p=2, dim=-1, keepdim=True)


def calculate_intrinstic_mean(data, iters=1, lr=1.000, init=None):
    """Calculate the intrinsic mean"""
    mean = data[0] if init is None else init  # init with first datapoint if not specified

    with torch.no_grad():
        for i in range(iters):
            grad = torch.mean(logarithmic_map(mean, data), dim=0)
            mean = exponential_map(mean, lr * grad).squeeze()
    return mean / torch.norm(mean, p=2)
