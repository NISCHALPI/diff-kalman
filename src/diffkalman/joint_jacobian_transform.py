"""Joint Jacobian Transformation to compute the Jacobian of a function with respect to its first argument and the function output."""

from torch.func import jacrev

__all__ = ["joint_jacobian_transform"]


def joint_jacobian_transform(f: callable) -> callable:
    """
    Transforms a given function f into a joint function that returns both the
    function's output and its Jacobian matrix with respect to its first argument.

    The transformed function, when called, returns a tuple where the first element
    is the Jacobian matrix of the function `f` with respect to its first argument
    and the second element is the output of the function `f`.

    Args:
        f (callable): The function to be transformed. It should be of the form
                      `f(x, *args)` where `x` is the variable with respect to
                      which the Jacobian is computed and `*args` are additional
                      arguments.

    Returns:
        callable: A new function `F` such that `F(x, *args)` returns a tuple
                  `(Df(x, *args), f(x, *args))`, where `Df(x, *args)` is the
                  Jacobian matrix of `f` with respect to `x` and `f(x, *args)`
                  is the original output of `f`.

    Example:
        >>> def my_function(x, a, b):
        >>>     return a * x + b
        >>>
        >>> F = joint_jacobian_transform(my_function)
        >>> x = torch.tensor([1.0, 2.0])
        >>> a = torch.tensor([2.0])
        >>> b = torch.tensor([3.0])
        >>> jacobian, output = F(x, a, b)
        >>> print(jacobian)  # Jacobian of my_function with respect to x
        >>> print(output)    # Output of my_function
    """

    # Define the new joint function
    def joint_func(*args, **kwargs):
        # Compute the output of the function
        output = f(*args, **kwargs)
        return output, output

    # The new joint jacrev function
    Df = jacrev(
        func=joint_func,
        argnums=0,
        has_aux=True,
    )
    return Df
