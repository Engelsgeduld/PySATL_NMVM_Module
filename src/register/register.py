from typing import Callable, Generic, Optional, Tuple, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Register of objects. Implementation of Register pattern"""

    def __init__(self, default: Optional[Type[T]] = None) -> None:
        """

        Args:
            default: Default object to return

        """
        self.default = default
        self.register_of_names: dict[Tuple[str, str], Type[T]] = {}

    def register(self, name: str, purpose: str) -> Callable:
        """Register new object

        Args:
            name: Class name
            purpose: Purpose of the algorithm ("NMParametric, NVParametric, NMVParametric,
                                                NMSemiparametric, NVSemiparametric, NMVSemiparametric")

        Returns: Decorator function

        Raises:
            ValueError: When class with this name was registered before

        """

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self.register_of_names:
                raise ValueError("This name is already registered")
            if purpose not in [
                "NMParametric",
                "NVParametric",
                "NMVParametric",
                "NMSemiparametric",
                "NVSemiparametric",
                "NMVSemiparametric",
            ]:
                raise ValueError("Unexpected purpose value")
            self.register_of_names[(name, purpose)] = cls
            return cls

        return decorator

    def dispatch(self, name: str, purpose: str) -> Type[T]:
        """Find object by name

        Args:
            name: Class name
            purpose: Purpose of the algorithm ("NMParametric, NVParametric, NMVParametric,
                                                NMSemiparametric, NVSemiparametric, NMVSemiparametric")

        Returns: object

        Raises:
            ValueError: When object with this name was not found

        """
        if (name, purpose) in self.register_of_names:
            return self.register_of_names[(name, purpose)]
        if self.default is None:
            raise ValueError(f"{name}, {purpose} realisation not registered")
        return self.default
