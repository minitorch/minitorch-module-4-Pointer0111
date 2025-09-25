from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """

    eps = 1e-5
    vals1 = list(vals)
    vals2 = list(vals)
    vals1[arg] += eps
    vals2[arg] -= eps
    return (f(*vals1) - f(*vals2)) / (2 * eps)





variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    if variable.is_constant():
        return []
    visited = set()
    order = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        order.append(v)
    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    对计算图进行反向传播，计算所有叶子节点的导数。

    参数：
        variable: 计算图最右侧的变量（输出节点）
        deriv: 该节点的初始导数（通常为1.0）

    实现思路：
    1. 用 derivatives 字典记录每个节点的导数，key 为 unique_id。
    2. 用 node_map 记录 unique_id 到节点对象的映射，方便后续访问。
    3. 按照拓扑排序（从输出到输入）遍历所有节点。
    4. 对于每个非叶子节点，使用 chain_rule 计算其所有父节点的导数，并累加到 derivatives。
    5. 最后只对叶子节点调用 accumulate_derivative，把最终导数写入变量。

    详细步骤：
    - 初始化：将输出节点 variable 的导数设为 deriv。
    - 遍历：对每个节点 v，若不是叶子节点，则用 v.chain_rule 计算其父节点的导数，并累加。
    - 终结：遍历所有节点，只对叶子节点调用 accumulate_derivative，把导数写入。
    """
    # 1. 维护每个节点的导数
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    node_map = {variable.unique_id: variable}
    # 2. 按照拓扑排序遍历所有节点
    for v in topological_sort(variable):
        node_map[v.unique_id] = v
        # 3. 跳过叶子节点（叶子节点不需要继续传播）
        if v.is_leaf():
            continue
        # 4. 获取当前节点的导数
        d_output = derivatives.get(v.unique_id, 0.0)
        # 5. 对每个父节点，累加导数
        for parent, parent_deriv in v.chain_rule(d_output):
            node_map[parent.unique_id] = parent
            if parent.unique_id in derivatives:
                derivatives[parent.unique_id] += parent_deriv
            else:
                derivatives[parent.unique_id] = parent_deriv
    # 6. 只对叶子节点累加导数
    for uid, v in node_map.items():
        if v.is_leaf() and uid in derivatives:
            v.accumulate_derivative(derivatives[uid])


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
